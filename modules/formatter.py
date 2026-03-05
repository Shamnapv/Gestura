def format_text(sentence,mode):

    words = sentence.upper().split()

    if mode=="Simplified":

        remove_words = {"IS","AM","ARE","THE","A","AN","PLEASE"}

        return " ".join([w for w in words if w not in remove_words])

    if mode=="Formal":

        if sentence.startswith("MY NAME"):
            return sentence + ". NICE TO MEET YOU."

        if sentence.startswith("WHAT"):
            return "COULD YOU PLEASE " + sentence.lower() + "?"

        return sentence + "."

    return sentence