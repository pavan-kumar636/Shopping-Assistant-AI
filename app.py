from flask import Flask, render_template, request
from watsonx_llm import refined_query, generate_comparision_table
from serp_api import search_products

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    user_input = request.form["query"]
    location = request.form.get("location", "India")

    refined_query_text = ""  # Initialize with a default value
    try:
        refined_query_response = refined_query(user_input, location)
        print(f"Refined query response: {refined_query_response}") # Log the refined query
        refined_query_text = f"{refined_query_response['refined_query']} {refined_query_response['additional_info']}".strip()
    except Exception as e:
        print(f"Error during refined_query: {e}")
        # Handle the error appropriately, maybe set a default refined_query_text
        refined_query_text = user_input  # Fallback to the original input

    products = search_products(refined_query_text, location=location)
    if not products:
        return render_template("index.html", error="No products found for your query.")

    comparision_table, summary = generate_comparision_table(products)
    print(f"Comparison table being passed to template: {comparision_table}") # Log before rendering
    print(f"Summary being passed to template: {summary}") # Log before rendering

    return render_template(
        "index.html",
        refined_query=refined_query_text,
        comparision_table=comparision_table,
        summary=summary
    )

'''
@app.route("/search", methods=["POST"])
def search():
    user_input = request.form["query"]
    location = request.form.get("location", "India")

    refined_query_response = refined_query(user_input, location)
    refined_query = f"{refined_query_response['refined_query']} {refined_query_response['additional_info']}".strip()

    products = search_products(refined_query, location=location)
    if not products:
        return render_template("index_html", error="No products found for your query.")

    comparision_table, summary = generate_comparision_table(products)

    return render_template(
        "index.html",
        refined_query=refined_query,
        comparision_table=comparision_table,
        summary=summary
    )
'''


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)