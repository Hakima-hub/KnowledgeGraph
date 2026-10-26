import json
import os
import re

import streamlit as st
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase

llm = ChatOllama(model="gemma3", temperature=0)

URI = os.getenv("URI", "neo4j://127.0.0.1:7687")
USER = os.getenv("USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j12345678")

STARTER_FACTS = [
    ("paris", "has attraction", "eiffel tower"),
    ("paris", "has attraction", "louvre museum"),
    ("paris", "famous for food", "croissant"),
    ("paris", "best time to visit", "spring"),
    ("paris", "local transport", "metro"),
    ("paris", "country", "france"),
    ("tokyo", "has attraction", "senso-ji temple"),
    ("tokyo", "has attraction", "shibuya crossing"),
    ("tokyo", "famous for food", "sushi"),
    ("tokyo", "famous for food", "ramen"),
    ("tokyo", "best time to visit", "spring"),
    ("tokyo", "local transport", "jr train"),
    ("tokyo", "country", "japan"),
    ("rome", "has attraction", "colosseum"),
    ("rome", "has attraction", "vatican city"),
    ("rome", "famous for food", "pasta"),
    ("rome", "best time to visit", "autumn"),
    ("rome", "local transport", "metro"),
    ("rome", "country", "italy"),
]

@st.cache_resource
def get_driver():
    return GraphDatabase.driver(URI, auth=(USER, NEO4J_PASSWORD))


def run_query(query, **params):
    """Run a query and return the rows as a list of dicts."""
    with get_driver().session() as session:
        return session.run(query, **params).data()


def add_facts(facts):
    """Save (subject, relation, object) facts into Neo4j."""
    for subject, relation, obj in facts:
        run_query(
            """
            MERGE (a:Entity {name: $subject})
            MERGE (b:Entity {name: $obj})
            MERGE (a)-[:RELATED {relation: $relation}]->(b)
            """,
            subject=subject, relation=relation, obj=obj,
        )


def database_is_empty():
    return run_query("MATCH (n:Entity) RETURN count(n) AS c")[0]["c"] == 0


def get_destinations():
    rows = run_query(
        """
        MATCH (a:Entity)-[:RELATED {relation: 'country'}]->()
        RETURN DISTINCT a.name AS name ORDER BY name
        """
    )
    return [row["name"] for row in rows]


def place_exists(name):
    return run_query(
        "MATCH (n:Entity {name: $name}) RETURN count(n) AS c", name=name
    )[0]["c"] > 0


def generate_facts(place):
    """Ask gemma3 for travel facts about a place."""
    prompt = f"""Give travel facts about {place} as triples.
Use ONLY these relations: has attraction, famous for food,
best time to visit, local transport, country.
Return ONLY a JSON list, like:
[["{place}", "has attraction", "..."], ["{place}", "country", "..."]]"""
    reply = llm.invoke(prompt).content
    match = re.search(r"\[.*\]", reply, re.S)
    try:
        triples = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        return []
    return [tuple(str(x).lower().strip() for x in t) for t in triples if len(t) == 3]


def find_place(question):
    """Ask gemma3 which city or country the question is about."""
    reply = llm.invoke(
        "Which city or country is this travel question about? "
        f"Reply with only the name, or NONE.\n\nQuestion: {question}"
    ).content
    return reply.strip().lower().strip(".")

def get_matches(question):
    """Find entities in the database whose name appears in the question."""
    rows = run_query(
        "MATCH (n:Entity) WHERE $question CONTAINS n.name RETURN n.name AS name",
        question=question.lower(),
    )
    return [row["name"] for row in rows]


def get_facts(matches):
    """Get every fact connected to the matched entities."""
    rows = run_query(
        """
        MATCH (a:Entity)-[r:RELATED]-(b:Entity)
        WHERE a.name IN $names
        RETURN DISTINCT startNode(r).name AS source,
                        r.relation AS relation,
                        endNode(r).name AS target
        """,
        names=matches,
    )
    return [(row["source"], row["relation"], row["target"]) for row in rows]

def clean(text):
    """Remove double quotes so they can't break the graph description."""
    return text.replace('"', "'")


def build_graph_text(facts, matches):
    lines = ["digraph {", "  rankdir=LR;", "  node [style=filled, fontcolor=white];"]
    seen = set()
    for source, relation, target in facts:
        for node in (source, target):
            if node not in seen:
                color = "#e45756" if node in matches else "#4c78a8"
                lines.append(f'  "{clean(node)}" [fillcolor="{color}"];')
                seen.add(node)
        lines.append(f'  "{clean(source)}" -> "{clean(target)}" [label="{clean(relation)}"];')
    lines.append("}")
    return "\n".join(lines)


st.set_page_config(page_title="Travel GraphRAG", page_icon="")
st.title(" Travel GraphRAG Assistant")

# Connect and load starter data on first run
try:
    if database_is_empty():
        add_facts(STARTER_FACTS)
except Exception as error:
    st.error(f"Could not connect to Neo4j: {error}")
    st.stop()

# Sidebar: add a destination by hand
st.sidebar.header("Add a destination")
new_place = st.sidebar.text_input("e.g. Bali")
if st.sidebar.button("Add to graph") and new_place:
    with st.spinner(f"Learning about {new_place}..."):
        new_facts = generate_facts(new_place.lower().strip())
        add_facts(new_facts)
    if new_facts:
        st.sidebar.success(f"Added {len(new_facts)} facts about {new_place}")
    else:
        st.sidebar.warning("Couldn't get facts, try again.")

destinations = get_destinations()
st.caption(f"{len(destinations)} destinations: " + ", ".join(destinations))

question = st.chat_input("Ask a travel question, e.g. What should I eat in Tokyo?")
if question:
    matches = get_matches(question)

    if not matches:
        place = find_place(question)
        if place != "none":
            if not place_exists(place):
                with st.spinner(f"Learning about {place}..."):
                    add_facts(generate_facts(place))
            matches = get_matches(question + " " + place)

    if not matches:
        st.warning("I couldn't work out which place you mean. Try naming a city or country.")
    else:
        facts = get_facts(matches)
        facts_text = "\n".join(f"{s} - {r} - {t}" for s, r, t in facts)
        answer = llm.invoke(
            f"""You are a friendly travel assistant.
Answer the question using only these facts:
{facts_text}

Question: {question}"""
        ).content

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Graph")
        st.graphviz_chart(build_graph_text(facts, matches))