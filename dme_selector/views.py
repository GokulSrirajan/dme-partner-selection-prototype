import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

# Initialize model
llm_model = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Prompt template
template = """
You are a DME logistics expert. Given the medical order and the list of DME partners,
evaluate each partner and produce an output.

Selection Logic:
- The best partner must be able to fulfill all the requested products. If not fulfilled, don't recommend a best partner.
- If a best partner exists, always list other partners who also fulfill all products as alternatives.
- If there is a best partner, do NOT suggest split_delivery.
- Use partner_rating, previous_delivery_satisfaction_rating, and contract quality as tie-breakers.
- If no single partner can fulfill all products, provide a 'split_delivery' — a set of partners that collectively fulfill the full order.
- Do not consider the 'status' field.

Output Format:
Respond ONLY in JSON with the following format:

{{
  "best_partner": {{
    "partner_id": "<ID>",
    "partner_name": "<Name>",
    "summary": "<Short explanation of the selection made>"
  }},
  "alternatives": [
    {{
      "partner_id": "<ID>",
      "partner_name": "<Name>",
      "summary": "<Short explanation of the selection made>"
    }}
  ],
  "split_delivery": [
    {{
      "partner_id": "<ID>",
      "partner_name": "<Name>",
      "fulfilled_products": ["<Product Name 1>", "<Product Name 2>"]
    }}
  ],
  "summary": "<Short explanation of the selection made>"
}}

Instructions:
- "summary" should be a one-line explanation describing why the selected partner or split was chosen.
- "best_partner" must fulfill all requested products.
- "alternatives" are backup partners that also fulfill all products but scored lower.
- Use "split_delivery" ONLY if no single partner can fulfill the full order — include product names each can supply.
- All output must be strictly in JSON. Do not explain outside the JSON block.

Medical Order:
{order}

DME Partners:
{partners}
"""

prompt = PromptTemplate.from_template(template)


def partner_matches_all_products(order_products, partner_catalog):
    for product in order_products:
        required_hcpcs = product.get("hcpcs_code", "").strip().upper()
        required_option = product.get("protocol_step_option", "").strip().lower()

        match_found = any(
            p.get("hcpcs_code", "").strip().upper() == required_hcpcs or
            p.get("protocol_step_option", "").strip().lower() == required_option
            for p in partner_catalog
        )
        if not match_found:
            return False
    return True


@csrf_exempt
def select_dme_partner(request):
    if request.method != 'POST':
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body = json.loads(request.body)
        order_json = body["order"]
        state = order_json["practice_details"]["address"]["address_state"]
        order_products = order_json["details"][0]["products"]

        # Load partners
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dme_path = os.path.normpath(os.path.join(base_dir, "../data/dme_partners.json"))
        with open(dme_path, "r") as f:
            dme_partners = json.load(f)

        # Filter based on state and payor
        eligible_partners = []
        for partner in dme_partners:
            if partner.get("state") != state:
                continue
            if not partner.get("contracted_payor_status", False):
                continue
            eligible_partners.append(partner)

        if not eligible_partners:
            return JsonResponse({"error": "No eligible DME partners found for the patient's state and payor status."}, status=404)

        # Format the prompt
        formatted_prompt = prompt.format(
            order=json.dumps(order_json, indent=2),
            partners=json.dumps(eligible_partners, indent=2)
        )

        # Get LLM response
        response = llm_model.invoke(formatted_prompt)
        content = response.content.strip()

        # Clean LLM output if it's wrapped in markdown
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.startswith("json\n"):
                content = content[len("json\n"):]
            elif content.startswith("json\r\n"):
                content = content[len("json\r\n"):]

        result_json = json.loads(content)

        # ✅ Post-processing: verify best_partner and split_delivery
        def product_names_from_order(order_products):
            return set(p.get("product_name", "").strip() for p in order_products)

        def product_names_from_partner(partner):
            return set(p.get("product_name", "").strip() for p in partner.get("product_catalog", []))

        requested_product_names = product_names_from_order(order_products)

        # Verify best partner
        if result_json.get("best_partner"):
            best_id = result_json["best_partner"]["partner_id"]
            best_partner = next((p for p in dme_partners if p["partner_id"] == best_id), None)
            if not best_partner or not requested_product_names.issubset(product_names_from_partner(best_partner)):
                result_json["best_partner"] = None

        # Verify alternatives
        filtered_alternatives = []
        for alt in result_json.get("alternatives", []):
            alt_id = alt["partner_id"]
            alt_partner = next((p for p in dme_partners if p["partner_id"] == alt_id), None)
            if alt_partner and requested_product_names.issubset(product_names_from_partner(alt_partner)):
                filtered_alternatives.append(alt)
        result_json["alternatives"] = filtered_alternatives

        # Verify split delivery
        if result_json.get("split_delivery"):
            combined_products = set()
            for part in result_json["split_delivery"]:
                fulfilled = set(part.get("fulfilled_products", []))
                combined_products.update(fulfilled)
            if not requested_product_names.issubset(combined_products):
                result_json["split_delivery"] = []
            # Also: if best_partner exists, clear split_delivery
            if result_json.get("best_partner"):
                result_json["split_delivery"] = []

        return JsonResponse(result_json)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": f"Invalid JSON from LLM: {str(e)}"}, status=500)