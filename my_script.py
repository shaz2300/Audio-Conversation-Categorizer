from collections import OrderedDict
import spacy
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')

class TextRank4Keyword():    
    def __init__(self):
        self.d = 0.85 
        self.min_diff = 1e-5 
        self.steps = 10 
        self.node_weight = None 
    
    def set_stopwords(self, stopwords):  
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            if(key in pur):
                d[1]+=value
            if(key in tdrive):
                d[2]+=value
            if(key in breakdown):
                d[3]+=value
            if(key in feedback):
                d[4]+=value
            if(key in vqual):
                d[5]+=value
            if(i > number):
                break
        
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN','VERB'], 
                window_size=4, lower=False, stopwords=list()):
        
       
        self.set_stopwords(stopwords)
        
        doc = nlp(text)
        
        sentences = self.sentence_segment(doc, candidate_pos, lower) 
        
        vocab = self.get_vocab(sentences)
        
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        g = self.get_matrix(vocab, token_pairs)
      
        pr = np.array([1] * len(vocab))
        
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight


import speech_recognition as sr

r = sr.Recognizer()
audio='Recording2.wav' #keep your own audio file in wav format in the variable -audio- by replacing -Recordin2.wav-

with sr.AudioFile(audio) as source:
    audio = r.record(source)

text = r.recognize_google(audio)

#the variable -text- contains your audio conversation in the form of text so you can print this to see the output of your speech to text conversion.

tr4w = TextRank4Keyword()
tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN','VERB'], window_size=4, lower=False)
pur=['purchase','enquiry','enquiries','location','purchasing','model','models','recent','cost','price','about','available','availability','affordable','closest','showroom','near','nearest','collection','brand','new','costs','costing','current','newly','modern','exchange','exchanging','exchanged','exchanges','existing','exist','futuristic','features','feature','characteristics','characters','properties','property','inquire','inquiries','query','ask','question','doubt','know','latest','latest','future','later','upcoming','prices','product']
tdrive=['appointment','make','cancel','available','book','appointments','booking','reschedule','car','scooty','scooter','vehicle','confirmation','confirm','confirmed','schedule','scheduling','test','drive','request','delayed','delay','delays','scheduled','try']
breakdown=['breakdown','brokendown','broken','car','location','vehicle','scooty','start','starting','won\'t','wouldn\t','assistance','failure','failing','fail','break','call','centre','center','contact','mechanic','repair','mechanics','report','reporting','reports','road','send','sending','stopped','stop','stopping','not']
feedback=['feedback','feedbacks','satisfied','unsatisfied','satisfactory','good','survey','surveys','after','service','post','services','customer','sales','sale','experience','experiences','response','conducting','conduct']
vqual=['attribute','quality','qualities','calibre','complain','complaint','complaints','condition','form','function','functioning','level','not','well','working','properly','status','worth']
#the above lists contain the possible terms used in the conversaton for that particular department
#for example- the list -pur- has all the words that might be used in the conversation of a -New vehicle purchase enquiries- department

d={1:0,2:0,3:0,4:0,5:0}

tr4w.get_keywords(500)

max=0
for i in d:
    if(d[i]>max):
        max=d[i]
        k=i

cat={1:"New vehicle purchase enquiries",2:"Test drive requests",3:"Breakdown",4:"Feedback",5:"Vehicle Quality"}
print(cat[k])

