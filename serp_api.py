from serpapi import GoogleSearch
from watsonx_llm import llm_generate_gl

def search_products(query, location):
    gl = llm_generate_gl(location)
    if not gl:
        print(f"Invalid or missing GL code for location '{location}'. Defaulting to 'INDIA'.")
        gl = "IN"
    print(f"Generated country code (gl): {gl}")
    params = {
        "q" : query,
        "tbm" : "shop",
        "location": location,
        "hl": "en",
        "gl": gl,
        "api_key" : "YOUR SERP-API KEY"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    print(f"SerpAPI Results: {results}")
    return results.get("shopping_results", [])

if __name__ == "__main__":
    test_query = "Buy affordable jeans online"
    test_location = "Austin, Texas, United States"
    products = search_products(test_query, location=test_location)
    for product in products:
        print(f"Title: {product['title']}")
        print(f"Price: {product['price']}")
        print(f"Source: {product['source']}")
        print(f"Link: {product['link']}")
        print(f"Rating: {product.get('rating', 'N/A')}")
        print(f"Reviews: {product.get('reviews', 'N/A')}")
        print("-" * 40)