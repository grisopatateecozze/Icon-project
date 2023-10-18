from owlready2 import *

print("ONTOLOGIA\n")
onto = get_ontology("Ontologia.owl").load()

# stampa il contenuto principale dell'ontologia
print("Class list in ontology-------------------------------------------------------------------------------------->\n")
print(list(onto.classes()), "\n")

# stampa le proprietà dell'oggetto
print("Object property in ontology--------------------------------------------------------------------------------->\n")
print(list(onto.object_properties()), "\n")

# stampa le proprietà dei dati
print("Object property in ontology--------------------------------------------------------------------------------->\n")
print(list(onto.data_properties()), "\n")

# stampa gli individui della classe paziente
print("paziente list in ontology----------------------------------------------------------------------------------->\n")
paziente = onto.search(is_a=onto.paziente)
print(paziente, "\n")

# stampa gli individui della classe domanda
print("individuals geni list in ontology--------------------------------------------------------------------------->\n")
geni = onto.search(is_a=onto.geni)
print(geni, "\n")

# QUERY
print("_____________________________QUERY_____________________________________________________________________________")

# Query per estrarre i pazienti con glioma, di sesso femminile e che presentino il gene ATRX mutato ma non quello BCOR
query_result = list(onto.search(type=onto.paziente, Gender=1, Grade=1, ATRX=1, BCOR=0))
print("\n- Pazienti di sesso FEMMINILE che sono affetti da LGG e presentano ATRX mutato e BCOR non mutato:\n")
for paziente in query_result:
    print(paziente.name)

# Query per estrarre i pazienti con glioma, di sesso maschile e che presentino il gene ATRX mutato ma non quello BCOR
query_result = list(onto.search(type=onto.paziente, Gender=0, Grade=1, ATRX=1, BCOR=0))
print("\n- Pazienti di sesso MASCHILE che sono affetti da LGG e presentano ATRX mutato e BCOR non mutato:\n")
for paziente in query_result:
    print(paziente.name)

# query per estrarre i pazienti con glioma, di sesso femminile e con età alla diagnosi < 30 anni
pazienti_femmine_LGG_30_plus = \
    [p for p in onto.paziente.instances() if p.Age_of_diagnosis and int(p.Age_of_diagnosis[0]) > 30 and 1 in p.Grade and 1 in p.Gender]
print("\n- Pazienti femmine affette da LGG con età > 30:\n")
for paziente in pazienti_femmine_LGG_30_plus:
    print(paziente)

# query per estrarre i pazienti con glioma, di sesso maschile e con età alla diagnosi < 30 anni
pazienti_femmine_LGG_30_plus = \
    [p for p in onto.paziente.instances() if p.Age_of_diagnosis and int(p.Age_of_diagnosis[0]) > 30 and 1 in p.Grade and 0 in p.Gender]
print("\n- Pazienti marchi affetti da LGG con età > 30:\n")
for paziente in pazienti_femmine_LGG_30_plus:
    print(paziente)
