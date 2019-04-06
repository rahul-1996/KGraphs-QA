from py2neo import Graph


graph = Graph("http://localhost:7474", username="neo4j", password="mamathahr")
#results= graph.run("MATCH (Test)-[:indicates_disease]-(x) WHERE Test.name =~ '(?i).*ekg.*' RETURN Test, x").to_data_frame()

#print(results["Test"][0]["name"])

graph.run('''LOAD CSV WITH HEADERS FROM "file:///conditions.csv" AS line  MERGE (patient:Patient{Name:line.PATIENT}) MERGE (problem:Problem{name:line.DESCRIPTION}) MERGE (patient)-[:has_disease{start_date:line.START , end_date:coalesce(line.STOP,"Unknown")}]-(problem)''')
graph.run('''LOAD CSV WITH HEADERS FROM "file:///medications.csv" AS line MERGE (patient:Patient{Name:line.PATIENT}) MERGE (medicine:Treatment{name:line.DESCRIPTION}) MERGE (disease:Problem{name:coalesce(line.REASONDESCRIPTION,"Unknown")}) MERGE (patient)-[:takes_medicine{start_date:line.START , end_date:coalesce(line.STOP,"Unknown"),  disease_name:coalesce(line.REASONDESCRIPTION,"Unknown")}]-(medicine) MERGE (medicine)-[:treats_disease]-(disease)''')
graph.run('''LOAD CSV WITH HEADERS FROM "file:///symptomDisease.csv" AS line MERGE (symptom:Symptom{name:line.symptom_name}) MERGE (disease:Problem{name:line.disease_name}) MERGE (disease)-[:has_symptom]-(symptom) ''')


