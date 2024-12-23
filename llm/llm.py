import openai


class MyLLM:
    def __init__(self) -> None:
        self.client = openai.OpenAI(
            api_key="sk-*",
            base_url='*'
        )
    
    
    def triplet2Query(self, prompt_text, model="gpt-3.5-turbo-0125"):
        conversation = list()
        conversation.append(
            {'role': 'user',
            "content": prompt_text})
        response = self.client.chat.completions.create(
            model=model,  #选择的GPT模型名称
            messages=conversation
        )
        result = response.choices[0].message.content
        return result
    def neighbors2text(self, neighbors, limitation=20, model="gpt-3.5-turbo-0125"):
        info = list()
        for i,triplet in enumerate(neighbors):
            if i == limitation:
                break
            h, r, t = triplet.split('\t')
            prompt = "Please convert the triple <{h},{r},{t}> into a natural language statement.".format(h=h,r=r,t=t)
            conversation = list()
            conversation.append(
                {'role': 'user',
                "content": prompt})
            response = self.client.chat.completions.create(
                model=model,  #选择的GPT模型名称
                messages=conversation
            )
            txt = response.choices[0].message.content
            info.append(txt)
        return info

    def construct_demon(self):
        demon = "Below is a demonstration example:\n**Knowledge from the knowledge graph**: ['The Plastic Ono Band is an avant-garde music group known for their work in the artists genre.', 'M.I.A. is an avant-garde artist known for her genre-bending music.', 'John Frusciante is an artist known for his avant-garde music in the genre of artists.', 'Avant-garde artists genre music is associated with Mike Patton.', 'Pink Floyd is an avant-garde music group within the artists genre.', 'The Fall is a music group known for their avant-garde style within the artists genre.', 'David Bowie is an avant-garde artist known for his contributions to the genre of music.', 'Sonic Youth is an avant-garde music group known for their experimental and innovative approach to the artists genre.', 'Buckethead is an avant-garde artist known for his genre-bending music.', 'Nico is an avant-garde musician in the genre of artists.', 'Les Claypool is an avant-garde musician known for his work in the artists genre of music.', 'Avant-garde music is a genre that is associated with artists like Jim O\'Rourke.', 'David Sylvian is an avant-garde artist known for his work in the genre of music.']\n**Knowledge from Wiki knowledge base**: ' Fundación Cultural para la Sociedad Mexicana, A.C. (Cultural Foundation for the Mexican Society) is a Mexican civil association whose primary activity is the operation of radio stations. The radio stations owned by FCSM are noncommercial (social) stations with Radio Maria programming. This is noteworthy, as Mexican law restricts religious programming on radio and prohibits the ownership of broadcast outlets by religious associations.'.\n**Please briefly answer the question<'Which genre of music is Avant-garde a subgenre of?'> based on the knowledge from the two sources mentioned above.**\n**The answer is**: ['20th-century classical music']."
        return demon

    def queryLLM(self, query_triplet, kg, wiki=None, llm=None, demon=False, model="gpt-3.5-turbo-0125"):
        """
        query_triplet: str ----三元组转后的自然语言问句
        kg(neighbors_text): list()
        wiki(retrieval_doc_top): list() {'id': '7122641', 'title': 'Fundación Cultural para la Sociedad Mexicana', 'section': '', 'text': ' Fundación...'}
        llm: 
        """
        prompt_kg = f"**Knowledge from the knowledge graph**: {str(kg)}.\n"
        prompt_wiki = ""
        if wiki:
            wiki_text = [wiki[i]["text"] for i in range(len(wiki))]
            prompt_wiki = f"**Knowledge from Wiki knowledge base**: {str(wiki_text)}.\n"

        prompt_back = f"Please combine the information of the two sources mentioned above, and briefly answer the question<{query_triplet}> according to all the knowledge you know now. Please help me generate ten candidate answers entities, and answer in strict accordance with the format of <The Answer is: [Answer1, Answer2, ..., Answer10]>."
        
        if demon:
            prompt_demon = self.construct_demon()
            prompt = prompt_kg + prompt_wiki + prompt_back + prompt_demon
        else:
            prompt = prompt_kg + prompt_wiki + prompt_back
        conversation = list()
        conversation.append(
            {'role': 'user',
            "content": prompt})
        response = self.client.chat.completions.create(
            model=model,  #选择的GPT模型名称
            messages=conversation
        )
        result = response.choices[0].message.content
        return result, prompt