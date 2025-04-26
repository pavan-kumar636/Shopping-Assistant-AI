from langchain_ibm import WatsonxLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import pandas as pd

MODEL_ID = "meta-llama/llama-3-1-70b-instruct"
WATSONX_URL = "Your Watsonx-AI Location URL"
PROJECT_ID = "Your Watsonx-AI Project ID"

model_parameters = {
    "decoding_method" : "greedy",
    "max_new_tokens" : 6000,
    "repetition_penalty" : 1
}

llm = WatsonxLLM(
    model_id = MODEL_ID,
    url = WATSONX_URL,
    project_id = PROJECT_ID,
    params = model_parameters,
    apikey = "YOUR WATSONX-AI API KEY"
)

class SerpAPIPromptResponse(BaseModel):
    refined_query: str = Field(description="Refined search query for the product search.")
    additional_info: str = Field(description="Additional adjectives summarized to be added to the search query.")

json_parser = JsonOutputParser(pydantic_object=SerpAPIPromptResponse)

llama_template = PromptTemplate(
    template="System:{system_prompt}\n{format_prompt}\nHuman: {user_prompt}\nAI:",
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)

def llm_generate_gl(location):
    system_prompt=(
        "You are an expert at identifying ISO 3166-1 alpha-2 country codes from locations. "
        "Output only the two-letter country code (e.g., 'IN' for India) for the provided location, "
        "without any additional text or explanations."
    )
    format_prompt = "Output the ISO 3166-1 alpha-2 country code only."
    user_prompt = f"Location:{location}"

    try:
        chain = llama_template | llm
        response = chain.invoke({
            "system_prompt": system_prompt,
            "format_prompt": format_prompt,
            "user_prompt": user_prompt
        })
        gl_code = response.strip()
        if len(gl_code) == 2 and gl_code.isalpha():
            return gl_code.upper()
        else:
            print(f"Invalid country code returned:{gl_code}")
            return None
    except Exception as e:
        print(f"Error in llm_generate_gl: {e}")
        return None

def refined_query(user_input, location):
    system_prompt = "You are a highly skilled shopping assistant. Refine user queries into specific product searches."
    format_prompt = json_parser.get_format_instructions()
    chain = llama_template | llm | json_parser
    response = chain.invoke({
        "system_prompt" : system_prompt,
        "format_prompt" : format_prompt,
        "user_prompt" : user_input
    })
    return response

def llm_generate_summary(comparison_df_llm):
    system_prompt = (
        "You are a highly skilled shopping assistant with expertise in comparing products and summarizing findings. "
        "Your task is to analyze product data and generate a detailed summary in valid HTML format."
    )
    format_prompt = (
        "Your response should include the following sections, each in valid HTML format:\n"
        "<h3>Best Value Product</h3>: Identify the product with the best price-to-value ratio. Explain your reasoning.\n"
        "<h3>Highest Rated Option</h3>: Highlight the product with the highest user rating and explain why it stands out.\n"
        "<h3>Unique Features</h3>: List any unique features of specific products and why they are beneficial.\n"
        "<h3>Trade-offs and Comparisons</h3>: Discuss trade-offs among the products, considering price, ratings, and reviews.\n"
        "<h3>Conclusion and Suggestion</h3>: Provide a recommendation for the best product or approach, with justification.\n"
        "If any section lacks sufficient data, explain the limitations explicitly but still include the section.\n"
        "Use <ul> for lists and <li> for each item where applicable."
    )
    user_prompt = (
        "Here is the product information in JSON format:\n"
        f"{comparison_df_llm.to_dict(orient='records')}"
    )

    try:
        chain = llama_template | llm
        response = chain.invoke({
            "system_prompt": system_prompt,
            "format_prompt": format_prompt,
            "user_prompt": user_prompt
        })
        summary = response.strip()

        required_sections = [
            "<h3>Best Value Product</h3>",
            "<h3>Highest Rated Option</h3>",
            "<h3>Unique Features</h3>",
            "<h3>Trade-offs and Comparisons</h3>",
            "<h3>Conclusion and Suggestion</h3>",
        ]
        for section in required_sections:
            if section not in summary:
                summary += f"\n{section}\n<p>Data unavailable for this section.</p>"
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "<h3>Summary Unavailable</h3><p>An error occurred while generating the summary. Please try again later.</p>"

def generate_comparision_table(products, featured_results=None, nearby_results=None):
    print(f"Generating comparison table for {len(products)} products.") # Log the input
    all_products = products[:5]
    product_summaries = [
        {
            "Name": f"<a href='{p.get('product_link', '#')}' target='_blank'>{p.get('title', 'N/A')}</a>",
            "Price Now": p.get("price", "N/A"),
            "Original Price": p.get("old_price", "N/A"),
            "Special Offer": ", ".join(p.get("extensions", [])) if p.get("extensions") else "No Discount",
            "Rating": f"{p.get('rating', 'N/A')} ⭐",
            "Reviews": f"{p.get('reviews', 'N/A')} Reviews",
            "Store": f"<img src='{p.get('source_icon', '#')}' alt='{p.get('source', 'N/A')}' style='height:20px;vertical-align:middle;'/> {p.get('source', 'N/A')}",
            "Delivery": p.get("delivery", "N/A"),
            "Image": f"<img src='{p.get('thumbnail', '#')}' alt='Product Image' style='height:50px;'/>"
        }
        for p in all_products
    ]
    product_summaries_llm = [
        {
            "Name": p.get("title", "N/A"),
            "Price Now": p.get("price", "N/A"),
            "Original Price": p.get("old_price", "N/A"),
            "Special Offer": ", ".join(p.get("extensions", [])) if p.get("extensions") else "No Discount",
            "Rating": f"{p.get('rating', 'N/A')}",
            "Reviews": f"{p.get('reviews', 'N/A')} Reviews",
            "Store": p.get("source", "N/A"),
            "Delivery": p.get("delivery", "N/A"),
        }
        for p in all_products
    ]

    comparision_df = pd.DataFrame(product_summaries)
    comparision_df_llm = pd.DataFrame(product_summaries_llm)

    if comparision_df.empty:
        print("No products found for comparison table.")
        return "<h3>No products Found</h3>","<h3>Summary Unavailable</h3><p>No product data available for analysis.</p>"
    
    print(f"DataFrame for LLM summary:\n{comparision_df_llm.to_string()}") # Log the dataframe sent to LLM
    summary = llm_generate_summary(comparision_df_llm)
    print(f"Generated summary: {summary}") # Log the generated summary

    comparision_table_html = comparision_df.to_html(index=False, escape=False)
    print(f"Generated comparison table HTML:\n{comparision_table_html}") # Log

    return comparision_table_html, summary

    if __name__ == "__main__":
        user_input = "Find affordable jeans in Austin"
        location = "Austin, Texas"
        refined_response = refined_query(user_input, location)
        print("Refined Query:", refined_response['refined_query'])
        print("Additional Info:", refined_response['additional_info'])

        sample_products = [
        {
            "title": "Levi's 501 Original Jeans",
            "price": "Rs.1995",
            "source": "Amazon",
            "link": "https://www.amazon.com/example1",
            "rating": 4.5,
            "reviews": 120
        },
        {
            "title": "Wrangler Men's Jeans",
            "price": "Rs.2990",
            "source": "Walmart",
            "link": "https://www.walmart.com/example2",
            "rating": 4.3,
            "reviews": 98
        },
        {
            "title": "Lee Relaxed Fit Jeans",
            "price": "Rs.2500",
            "source": "Target",
            "link": "https://www.target.com/example3",
            "rating": 4.1,
            "reviews": 85
        },
        {
            "title": "Gap Straight Jeans",
            "price": "Rs.1599",
            "source": "Gap",
            "link": "https://www.gap.com/example4",
            "rating": 4.4,
            "reviews": 110
        },
        {
            "title": "Uniqlo Slim Jeans",
            "price": "Rs.3900",
            "source": "Uniqlo",
            "link": "https://www.uniqlo.com/example5",
            "rating": 4.6,
            "reviews": 150
        }
    ]

    comparision_table, summary = generate_comparision_table(sample_products)
    print("Comparision Table:\n", comparision_table)
    print("summary:\n", summary)

