from eval import get_ner
from qa.GetIntent-WordNetSoln import get_Intent
from qa.ExecuteQueryGivenIntent import executeQuery

query = "Does cardiac arrest cause thyroid"
ner_result, ners = get_ner(query)
intent = get_Intent(ner_result)
print("The intent is" , intent))
res = executeQuery(intent, ners)
print(f"Res: {res}")

