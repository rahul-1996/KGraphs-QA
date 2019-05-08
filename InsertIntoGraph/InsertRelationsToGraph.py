from neo4jrestclient.client import GraphDatabase
from neo4jrestclient.query import Q
import re

# db = GraphDatabase("http://localhost:7474", username="neo4j", password="mamathahr")
 
db = GraphDatabase("http://ec2-54-224-85-24.compute-1.amazonaws.com:7474", username="neo4j", password="i-0264fccb248189ea5")
# # Create some nodes with labels
# user = db.labels.create("User")
# u1 = db.nodes.create(name="Marco")
# user.add(u1)
# u2 = db.nodes.create(name="Daniela")
# user.add(u2)
 
# beer = db.labels.create("Beer")
# b1 = db.nodes.create(name="Punk IPA")
# b2 = db.nodes.create(name="Hoegaarden Rosee")
# # You can associate a label with many nodes in one go
# beer.add(b1, b2)


# # User-likes->Beer relationships
# u1.relationships.create("likes", b1)
# u1.relationships.create("likes", b2)
# u2.relationships.create("likes", b1)
# # Bi-directional relationship?
# u1.relationships.create("friends", u2)


relation_path = "../../concept_assertion_relation_training_data/beth/rel"


def getEntityTypes(r):   
	if(r == 'TrCP'):
	    #Treatment caused Problem
	    entity1 = "Treatment"
	    entity2 = "Problem"
	elif(r=='TeCP'):
	    #Test to detect Problem (outcome not known)
	    entity1 = "Test"
	    entity2 = "Problem"
	elif(r == 'TrWP'):
	    #Treatment made problem worse
	    entity1 = "Treatment"
	    entity2 = "Problem"
	elif(r == "TeRP" ):
	    #A test revealed a problem (outcome known)
	    entity1 = "Test"
	    entity2 = "Problem"
	elif(r == "PIP"):
	    #One medical problem (symptom) reveals or vauses another medical problem (disease)
	    entity1= "Problem1"
	    entity2= "Problem2"
	elif(r == "TrAP"):
	    #Treatment given for a problem (Outcome not known)
	    entity1 = "Treatment"
	    entity2 = "Problem"
	elif(r == "TrNAP"):
	    #Treatment not adminstered/ stopped
	    entity1= "Treatment"
	    entity2 = "Problem"
	elif(r == "TrIP"):
	    #Treatment imporves problem (Outcome Known)
	    entity1 = "Treatment"
	    entity2 = "Problem"
	return entity1 , entity2

#Create Entities
treatment = db.labels.create("Treatment")
test = db.labels.create("Test")
problem = db.labels.create("Problem")

def get_entity(entityType):
	if(entityType == "Treatment"):
		entity = treatment
	elif(entityType == "Test"):
		entity = test
	else:
		entity = problem
	return entity

def createEntity(entityType, entity): 
	
	entity = re.sub('[^A-Za-z0-9%]+', '', entity)
	lookup = Q("name", istartswith=entity)
	node = db.nodes.filter(lookup)
	# print(node[0])
	if(len(node) >0):
		print("exists")
		e = node[0]
	else:
		print("does not exist")
		e = db.nodes.create(name=entity)
		entityType.add(e)
	return e

def getRelation(r):  
    if(r == 'TrCP'):
        #Treatment caused Problem
        return "treatment_caused"
    elif(r=='TeCP'):
        #Test to detect Problem (outcome not known)
        return "indicates_disease"
    elif(r == 'TrWP'):
        #Treatment made problem worse
        return "is_not_a_treatment"
    elif(r == "TeRP" ):
        #A test revealed a problem (outcome known)
        return "indicates_disease"
    elif(r == "PIP"):
        #One medical problem (symptom) reveals or vauses another medical problem (disease)
        return "has_symptom"
    elif(r == "TrAP"):
        #Treatment given for a problem (Outcome not known)
        return "treats_disease"
    elif(r == "TrNAP"):
        #Treatment not adminstered/ stopped
        return "is_not_a_treatment"
    elif(r == "TrIP"):
        #Treatment imporves problem (Outcome Known)
        return "treats_disease"

import os
import pdb
files = os.listdir(relation_path)
count = 0
for file in files:
    out=[]
    path = relation_path + "/" + file
    lines = open(path).read().strip().split('\n')
    for line in lines:
        print(line)
        if len(line.strip()) == 0 :
            break
        entity1 = line.split("\"")[1]
        relation = line.split("\"")[3]
        entity2 = line.split("\"")[5]
        entityTypes = getEntityTypes(relation)
        dbEntity1 = get_entity(entityTypes[0])
        e1 = createEntity(dbEntity1 , entity1)
        dbEntity2 = get_entity(entityTypes[1])
        e2 = createEntity(dbEntity2 , entity2)
        r = getRelation(relation)
        e1.relationships.create(r, e2)
	

