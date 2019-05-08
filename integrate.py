from eval import get_ner
from intent import get_intent
from execute import executeQuery

query = "Does cardiac arrest cause thyroid"
while(1):
	print("Please enter your question:\n")
	query = str(input())
	ner_result, ners = get_ner(query)
	print("Named entities:", ners)
	print("Intent query:",ner_result)
	intent = get_intent(("").join(ner_result))
	print("The intent is" , intent)
	res = executeQuery(intent, ners)

