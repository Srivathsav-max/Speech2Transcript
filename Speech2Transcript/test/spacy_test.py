import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("Asifa, a care manager from Dr. Cameron’s office, called Mr. Smith to check on his health. She greeted him warmly and asked about any unusual symptoms, to which Mr. Smith responded that he was doing fine. She inquired about his medications, and he confirmed that he was taking them as prescribed, including his blood sugar and blood pressure medications, though he mentioned some recent changes. Asifa asked if he was experiencing any side effects, and he reassured her that he had none. She also checked if he had been to the emergency room or hospitalized recently, and he confirmed that he had only been to the clinic. When asked about changes in his daily routine, Mr. Smith shared that his routine remained the same, consisting of coffee, TV, spending time with his grandkids, and walking his dog. He also mentioned planning a trip to Hawaii in the summer."

"Asifa then asked if he had any health concerns, and Mr. Smith said he was doing well with no major issues. They discussed his diabetes management, and while he admitted to checking his blood glucose levels only once or twice a week instead of daily, he felt his diet was balanced and that he no longer needed insulin due to weight loss. He mentioned that Dr. Cameron had approved the change, especially after he started using Ozempic. He estimated his blood glucose levels to be around 100-110 in the mornings. Asifa reassured him that anything under 126 was good and asked about his diet. Mr. Smith said he had been mostly strict but occasionally indulged. He credited Ozempic for reducing his appetite and noted that he had lost 15 pounds in the last three months."

"When asked about physical activity, Mr. Smith confirmed that he walked regularly, often for 30-45 minutes, multiple times a day, and enjoyed chatting with neighbors during his walks. Asifa then inquired about his blood pressure, and he admitted to checking it only once or twice a week rather than daily. However, he mentioned that his readings were typically around 120-130. Asifa advised him to check it more frequently due to the risks associated with diabetes and hypertension. Regarding salt intake, he said he had reduced processed food and was mindful of adding extra salt."

"Asifa also asked about his weight, and Mr. Smith confirmed his weight loss but mentioned that his weighing scale needed new batteries. She then inquired about smoking and alcohol consumption, and he stated that he had never smoked and only had an occasional glass of wine. Regarding vaccinations, he confirmed that Dr. Cameron kept him up to date on his pneumonia and flu vaccines. When asked about his colonoscopy, he said he had one last year and was up to date on screenings."

"Transportation was not an issue for Mr. Smith, as he could drive and had support from his children if needed. He also confirmed that he was not currently seeing any specialists like a cardiologist or endocrinologist. Asifa concluded the call by affirming that his blood sugar and blood pressure levels were stable and encouraging him to continue his current routine. She informed him that she would follow up in two to three weeks and reminded him to call if he needed anything. They exchanged goodbyes, and the conversation ended on a positive note.")
doc = nlp(text)

# Analyze syntax
# print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
# print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.sents:
    print(entity.text)