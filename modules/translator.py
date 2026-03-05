from googletrans import Translator

translator = Translator()

def translate_text(text,language):

    if language=="Malayalam":
        return translator.translate(text,dest="ml").text

    if language=="Hindi":
        return translator.translate(text,dest="hi").text

    return text