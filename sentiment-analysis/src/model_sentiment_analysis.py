
def tweeter_preprocessor(s):
    tweet_p.set_options(tweet_p.OPT.EMOJI,tweet_p.OPT.URL)
    s = tweet_p.clean(s)
    s = re.sub(r'\b(?:a*(?:ja)+h?|(?:l+o+)+l+)\b', ' ', s)
    s = re.sub(r'[^\w]', ' ', s)
    s = unidecode.unidecode(s)
    s = tweet_p.clean(s)

    return s
chunk.columns
