from py2neo import Graph
import pandas as pd


e = ""
graph = Graph("http://18.215.249.105:7474", username="neo4j", password="i-0264fccb248189ea5")

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
	a = intent.split("_")[0] 
	q = intent.split("_")[1] + "s"
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
