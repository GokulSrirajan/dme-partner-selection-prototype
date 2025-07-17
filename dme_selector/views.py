import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load .env and set OpenAI key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

# Init LangChain model
llm_model = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Prompt template
template = """
You are a DME logistics expert. Given the medical order and the list of DME partners,
evaluate each partner and produce an output.

Selection Logic:
- The best partner must be able to fulfill all the requested products if not fullfilled don't recommend a best partner.
- Even if there is a best partner suggest alternatives if available
- Use partner_rating, previous_delivery_satisfaction_rating, and contract quality as tie-breakers.
- If no single partner can fulfill all products, provide a 'split_delivery' — a set of partners that collectively fulfill the full order.

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
    }},
    "summary": "<Short explanation of the selection made>",
  ]
}}

Instructions:
- "summary" should be a one-line explanation describing why the selected partner (or set) was chosen.
- "best_partner" must fulfill all requested products.
- "alternatives" are backup partners that also fulfill all products but scored lower.
- Only include "split_delivery" if no single partner can fulfill the entire order — list the product names each can provide.
- All output must be strictly in JSON format. Do not explain outside the JSON block.

Medical Order:
{order}

DME Partners:
{partners}
"""

prompt = PromptTemplate.from_template(template)


def partner_matches_all_products(order_products, partner_catalog):
    """
    Checks if all products in the order are available in the partner's catalog
    by matching either hcpcs_code or protocol_step_option.
    """
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
        # Parse order input
        body = json.loads(request.body)
        order_json = body["order"]
        state = order_json["practice_details"]["address"]["address_state"]
        order_products = order_json["details"][0]["products"]

        # Load partner JSON file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dme_path = os.path.normpath(os.path.join(base_dir, "../data/dme_partners.json"))
        print(f"📁 Loading DME partners from: {dme_path}")

        with open(dme_path, "r") as f:
            dme_partners = json.load(f)

        # Filter partners
        eligible_partners = []
        for partner in dme_partners:
            if partner.get("state") != state:
                print(f"❌ Skipped {partner['partner_name']}: State mismatch")
                continue
            if not partner.get("contracted_payor_status", False):
                print(f"❌ Skipped {partner['partner_name']}: Not contracted")
                continue
            if not partner_matches_all_products(order_products, partner.get("product_catalog", [])):
                print(f"❌ Skipped {partner['partner_name']}: Missing product(s)")
                continue

            print(f"✅ Eligible: {partner['partner_name']}")
            eligible_partners.append(partner)

        if not eligible_partners:
            return JsonResponse({"error": "No eligible DME partners found after filtering."}, status=404)

        # Format prompt for LLM
        formatted_prompt = prompt.format(
            order=json.dumps(order_json, indent=2),
            partners=json.dumps(eligible_partners, indent=2),
            state=state
        )

        # Call LLM
        response = llm_model.invoke(formatted_prompt)
        content = response.content.strip()
        
        # Clean Markdown code block (e.g., ```json\n ... \n```)
        if content.startswith("```"):
            content = content.strip("`").strip()
            # Remove the leading language if exists
            if content.startswith("json\n"):
                content = content[len("json\n"):]
            elif content.startswith("json\r\n"):
                content = content[len("json\r\n"):]

        # Now it's clean JSON
        result_json = json.loads(content)

        return JsonResponse(result_json)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)