# Path: visualize_graph.py
"""
Neo4j -> pyvis graph exporter (fixed).
- Removes unsupported `notebook` kwarg to Network.show()
- Explicitly closes the Neo4j driver to avoid destructor warnings
- Handles empty results gracefully and prints helpful messages
Usage:
    python visualize_graph.py
Output:
    neo4j_viz.html (open in browser)
"""

from neo4j import GraphDatabase
from pyvis.network import Network
import config
import sys

NEO_BATCH = 500  # number of relationships to fetch / visualize

# create driver (will be closed explicitly)
driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))


def fetch_subgraph(tx, limit=500):
    q = (
        "MATCH (a:Entity)-[r]->(b:Entity) "
        "RETURN a.id AS a_id, labels(a) AS a_labels, a.name AS a_name, "
        "b.id AS b_id, labels(b) AS b_labels, b.name AS b_name, type(r) AS rel "
        "LIMIT $limit"
    )
    return list(tx.run(q, limit=limit))


def build_pyvis(rows, output_html="neo4j_viz.html"):
    if not rows:
        print("No relationships found in Neo4j. Nothing to visualize.")
        return

    net = Network(height="900px", width="100%", directed=True)
    # optional: better physics/layout
    net.barnes_hut()

    seen_nodes = set()
    for rec in rows:
        a_id = rec.get("a_id"); a_name = rec.get("a_name") or a_id
        b_id = rec.get("b_id"); b_name = rec.get("b_name") or b_id
        a_labels = rec.get("a_labels") or []
        b_labels = rec.get("b_labels") or []
        rel = rec.get("rel") or ""

        if a_id not in seen_nodes:
            net.add_node(a_id, label=f"{a_name}\n({','.join(a_labels)})", title=str(a_name))
            seen_nodes.add(a_id)
        if b_id not in seen_nodes:
            net.add_node(b_id, label=f"{b_name}\n({','.join(b_labels)})", title=str(b_name))
            seen_nodes.add(b_id)

        # add edge with relationship as title / label
        net.add_edge(a_id, b_id, title=str(rel), label=str(rel))

    # Note: pyvis.Network.show does not accept `notebook` in some versions.
    net.show(output_html)
    print(f"Saved visualization to {output_html}")


def main():
    rows = []
    try:
        with driver.session() as session:
            rows = session.execute_read(fetch_subgraph, limit=NEO_BATCH)
    except Exception as e:
        print("ERROR: Failed to fetch data from Neo4j:", e)
        # ensure driver is closed before exit
        try:
            driver.close()
        except Exception:
            pass
        sys.exit(1)

    try:
        build_pyvis(rows)
    except Exception as e:
        print("ERROR: Failed to build visualization:", e)
    finally:
        # explicit close to avoid destructor issues/warnings
        try:
            driver.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
