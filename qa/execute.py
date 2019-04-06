from py2neo import Graph
import pandas as pd


#results= graph.run("MATCH (Test)-[:indicates_disease]-(x) WHERE Test.name =~ '(?i).*ekg.*' RETURN Test, x").to_data_frame()

#print(results["Test"][0]["name"])
e = ""
graph = Graph("http://localhost:7474", username="neo4j", password="mamathahr")

def getQueryForIntent(intent, entity):
	d = pd.read_excel("QA.xlsx", sheetname='Intent-Query')

	query = list(d[d["Intent"] == intent]["Query"])[0].replace("entity_name", entity)
	return  query


def runQueryAndGetResult(query):
	results= graph.run(query).to_data_frame()
	# print(len(results))
	# print(results)
	answer=[]
	for i in range(0 , min(5, len(results))):
		answer.append(results["x"][i]["name"])
	return answer

def printAnswer(results , intent):
	q = intent.split("_")[0] + "s"
	a = intent.split("_")[1]
	ans = (", ").join(results)
	print("Some " ,  q , " for " , a , e, " are " , ans )


def executeQuery(intent , entity):
	done = False
	for x in entity:
		query = getQueryForIntent(intent , x)
		# print(query)
		results=runQueryAndGetResult(query)
		# print(x ,results)
		if(len(results)!=0):
			global e
			e = x
			done = True
			printAnswer(results , intent)
			break
	if(not done):
		print("Sorry i dont really have the answer to that!" )


if __name__ == '__main__':
	executeQuery("disease_symptom" , ["cardiac arrest" , "thyroid"])