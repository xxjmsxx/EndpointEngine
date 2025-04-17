import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

def get_graph_connection():
    uri = os.getenv("NEO4J_URL")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASS")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

def fetch_variable_and_value_nodes(graph):
    """Fetch variable and value nodes from Neo4j"""
    query = """
    MATCH (v:Variable)-[:BELONGS_TO]->(c:Category)
    OPTIONAL MATCH (v)-[:HAS_VALUE]->(val:Value)
    RETURN v.name AS var_name,
           v.description AS var_description,
           c.name AS category,
           val.label AS value_label
    """
    return graph.run(query).data()

def extract_variable_array_from_text(text: str):
    match = re.search(r"\[([^\]]+)\]", text)
    if not match:
        return []

    inner = match.group(1)
    variables = [v.strip().strip("\"'`") for v in inner.split(",") if v.strip()]
    return variables

# --- Retrieve all value labels grouped by variable name ---
def get_all_values_for_variables(graph, variable_names):
    values_by_variable = {}

    for var_name in variable_names:
        query = """
        MATCH (v:Variable {name: $var_name})
        OPTIONAL MATCH (v)-[:HAS_VALUE]->(val:Value)
        RETURN val.label AS label
        ORDER BY val.label
        """
        result = graph.run(query, var_name=var_name)

        values = [record["label"] for record in result if record.get("label")]
        values_by_variable[var_name] = values

    return values_by_variable
