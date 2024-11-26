

def construct_fact_counterspeech_prompt(hatespeech):
    prompt = f"""Given the following hate speech, list relevant facts to generate a counter-speech:
    Hate speech: {hatespeech}
    The counter-speech should be concise, suitable for posting on social media.
    
    Example:
    Hate speech: "People of a certain race are all criminals."
    
    Counter-speech: "In fact, according to FBI statistics, crime rates are not correlated with race, and crime rates have been declining across all races. Additionally, numerous studies indicate that socioeconomic factors and education levels are the main determinants of crime rates. Many individuals and organizations within communities are actively working to promote fairness and safety in society.
    
    Give me the counterspeech only and as following format:
    [Counterspeech]: 
    """
    return prompt


def construct_vote_prompt(hatespeech, claims):
    claimstr = """"""
    for i in range(len(claims)):
        claimstr = claimstr + "[" + str(i) + "]" + claims[i] + "\n"
    vote_prompt = f'''Given an hatspeech: "{hatespeech}" and several counter-claims, decide one that can most effectively counter the hate speech. Conclude in the last line "The best claim is [s]", where s the integer id of the choice.

    claims: 
    {claimstr}
    '''
    return vote_prompt


def construct_claim_prompt(hatespeech,claim_num):
    claim_prompt = f"""
    Generate {claim_num} claims to refute the statement:{hatespeech}
    These claims should be well-structured and provide a strong foundation for retrieving factual knowledge to support them.
    You only need to generate short, direct claims without providing facts.
    List each counter-claim as follows:
    [claims]:
    1. [First counter-claim]
    2. [Second counter-claim]
    3. [Third counter-claim]
    """
    return claim_prompt


def construct_query_prompt(hatespeech, claim, query_num):
    query_prompt = f"""
    Given a hateful statement and a corresponding counter-claim, your task is to generate {query_num} search queries to retrieve evidence from Wikipedia that supports the counter-claim. 
    The query should be precise and relevant to ensure the retrieval of strong factual evidence.
    Here is the hateful statement:
    "{hatespeech}"
    Here is the counter-claim:
    "{claim}"
    Please generate a search query for Wikipedia to find evidence supporting the counter-claim.
    List each query as follows:
    [queries]:
    1. [the First query]
    2. [the Second query]
    3. [the Third query]
    4. [the Second query]
    5. [the Third query]
    """
    return query_prompt


def construct_argumentation_prompt(claim,evidence):
    evidence_str = ""
    for i in evidence:
        evidence_str = evidence_str + i + "\n"
    prompt = f"""Given a claims and relevant evidence, your task is to generate a statement. 
    The statement should be concise, respectful, and based on the provided evidence to effectively support the claim.
    The statement consists of claim and evidence
    Here is the claim:
    {claim}
    Here are the evidence:
    {evidence_str}
    Give me the argumentation as following format:
    [Argumentation]: ""
    """
    return prompt


def construct_claim_and_evidence(claims,evidence_list):
    prompt = """ """
    for i in range(len(claims)):
        prompt_claim = f"""
        Claim_{i}:{claims[i]}
        """
        print(prompt_claim)
        prompt_evidence = """"""
        for j in range(len(evidence_list[i])):
            prompt_evidence = prompt_evidence + f"""{evidence_list[i][j]}""" + "\n"
            print(prompt_evidence)
            prompt_single = prompt_claim + prompt_evidence
        prompt = prompt + prompt_single
    return prompt

def construct_counterspeech_prompt(hatespeech, claim,evidence):
    evidence_str = ""
    for i in range(len(evidence)):
        evidence_str = evidence_str + str(i) +". "+ evidence[i] + "\n"
    prompt = f"""You are a seasoned volunteer dedicated to countering hate speech on social media. 
    Given a claim and relevant evidence of each claim, your task is to generate a counterspeech. 
    The Counterspeech needs to first state the claim and then provide evidence to support the claim.
    The counterspeech should be effectively refute the hatespeech.
    You must give me the counterspeech as following format:
    [Counterspeech]: ""
    
    Here is the hate speech:
        "{hatespeech}"
    Here is the counter-claim:
        "{claim}"
    Here is the evidence:
        "{evidence_str}"

    """
    return prompt



