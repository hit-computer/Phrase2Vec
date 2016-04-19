#coding:utf-8
import argparse
import numpy as np
import pickle,time
import logging
logging.basicConfig(level=logging.DEBUG)

import theano
import theano.tensor as T
import collections,random,cPickle
from sklearn.metrics.pairwise import cosine_similarity  

parser = argparse.ArgumentParser(description='Phrase2vec')

parser.add_argument('file', type=str,
                    help='the input file name')
parser.add_argument('--size', type=int, default=50,
                    help='the dimention of vector')                    
parser.add_argument('--min-wf', type=int, default=10,
                    help = 'word frenquency less than this value will be ignore')
parser.add_argument('--min-pf', type=int, default=10,
                    help = 'phrase frenquency less than this value will be ignore')
parser.add_argument('--margin', type=float, default=1.0,
                    help='between 0 and 1')
parser.add_argument('--window', type=int, default=5,
                    help='the size of window')
parser.add_argument('--max-num-epochs', type=int, default=10,
                    help='the maximal number of training epochs')
parser.add_argument('--lr', type=float, default=.005,
                    help='the initial learning rate')
parser.add_argument('--output', type=str, default='out',
                    help='the output file name')
parser.add_argument('--neg-sample', type=int, default=10,
                    help='number of negative sample')
parser.add_argument('--save-epochs', type=int, default=1,
                    help='save the model every this value')
args = parser.parse_args()

class Data:
    word_dic = {} #词表，包含unigram和bigram
    docLS = [] #文本list
    wd_index_list = [] #每个词用其索引号表示，每个元素是一个Document
    curpos = 0
    idx_to_item = {}
    
    @classmethod
    def init(cls):
        num_item = collections.defaultdict(int)
        with open(args.file, 'r') as fr:
            for line in fr:
                line = line.rstrip('\r\n')
                line = line.decode('utf-8')
                line = line.split()
                cls.docLS.append(line)
                #计算unigram
                for wd in line:
                    num_item[wd] += 1
                #计算bigram
                for i in range(len(line)-1):
                    num_item[(line[i],line[i+1])] += 1
        #num_item_sorted = sorted(num_item.iteritems(),key=lambda x:x[1], reverse=True)
        idx = 0
        for item in num_item.iteritems():
            if type(item[0])==tuple and item[1] >= args.min_pf:
                cls.word_dic[item[0]] = (idx,item[1])
                idx += 1
            elif type(item[0])==unicode and item[1] >= args.min_wf:
                cls.word_dic[item[0]] = (idx,item[1])
                idx += 1
                
        for doc in cls.docLS:
            words = [cls.word_dic[i][0] for i in doc if i in cls.word_dic]
            if len(words) > 0:
                cls.wd_index_list.append(words)
            
        cls.idx_to_item = dict([(i[1][0],i[0]) for i in cls.word_dic.iteritems()])
    
    @classmethod
    def reset(cls):
        cls.curpos = 0
    @classmethod    
    def next(cls):
        if cls.curpos >= len(cls.wd_index_list):
            return None
        doc = cls.wd_index_list[cls.curpos]
        cls.curpos += 1
        #print 'curpos:',cls.curpos,len(cls.wd_index_list)
        return doc
    
    @classmethod
    def get_sizeofvcob(cls):
        return len(cls.word_dic)
    @classmethod
    def get_index(cls, item):
        return cls.word_dic[item][0]
    @classmethod
    def get_item(cls, idx):
        return cls.idx_to_item[idx]
    
    @classmethod
    def test(cls):
        #for key in cls.word_dic:
            #print key, cls.word_dic[key]
        for doc in cls.wd_index_list:
            print doc

class Updata: #该版本尚未使用Adam参数更新方法
    def __init__(self, param):
        #self.i = np.float32(0.)
        self.m = np.float32(param * 0.)
        self.v = np.float32(param * 0.)
       
    def Adam(self, param, grad, i_t, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    #updates = []
        #i_t = self.i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (np.sqrt(fix2) / fix1)
        m_t = (b1 * grad) + ((1. - b1) * self.m)
        v_t = (b2 * grad**2) + ((1. - b2) * self.v)
        g_t = m_t / (np.sqrt(v_t) + e)
        p_t = param - (lr_t * g_t)
        self.m = m_t
        self.v = v_t
        return p_t
            
if __name__ == '__main__':
    Data.init()
    logging.info('initial complete!')
    len_vcob = Data.get_sizeofvcob()
    print 'len_vcob:', len_vcob
    print 'num_of_doc', len(Data.wd_index_list)
    items_emb = np.random.uniform(size=(len_vcob,args.size))
    items_emb = np.float32(items_emb)
    i_t = np.float32(0.)
    
    T_context = T.fvector('context')
    T_x = T.fvector('phrase')
    T_neg = T.fmatrix('neg_sample')
    
    f_x = T.sum(T_context*T_x)
    cost = T.maximum(0, args.margin - f_x + T.dot(T_neg, T_context.dimshuffle(0,'x')))
    cost = cost.sum()
    
    x_grad = T.grad(cost, T_x)
    cont_grad = T.grad(cost, T_context)
    neg_grad = T.grad(cost, T_neg)
    
    f_grad = theano.function([T_x, T_context, T_neg],[cost,x_grad,cont_grad,neg_grad])
    logging.info('function build finish!')
    
    for iter in range(args.max_num_epochs):
        Data.reset()
        doc = Data.next()
        print 'iter:',iter
        cost_list = []
        stime = time.time()
        while doc:
            i_t += 1
            #print i_t
            for pos in range(len(doc)-1):
                context = np.zeros(args.size, dtype='float32')
                cont_len = 0
                pitem = (Data.get_item(doc[pos]), Data.get_item(doc[pos+1]))
                if not Data.word_dic.has_key(pitem):
                    cur_pos = doc[pos]
                    cur_vec = items_emb[cur_pos]
                    contLS = []
                    for c in range(pos-args.window, pos+args.window+1):
                        if c < 0:
                            continue
                        if c >= len(doc):
                            continue
                        if c == pos:
                            continue
                        wd_ = doc[c]    
                        context += items_emb[wd_]
                        cont_len += 1
                        contLS.append(wd_)
                else:
                    cur_pos = Data.get_index(pitem)
                    cur_vec = items_emb[cur_pos]
                    contLS = []
                    for c in range(pos-args.window, pos+args.window+2):
                        if c < 0:
                            continue
                        if c >= len(doc):
                            continue
                        if c == pos or c == pos+1:
                            continue
                        wd_ = doc[c]    
                        context += items_emb[wd_]
                        cont_len += 1
                        contLS.append(wd_)
                if cont_len == 0:
                    continue
                context = context/cont_len
                
                _range = range(len_vcob)
                _range.pop(cur_pos)
                nsample = random.sample(_range, args.neg_sample)
                neg_matrix = items_emb[nsample]
                
                cost_, x_grad_, cont_grad_, neg_grad_ = f_grad(cur_vec, context, neg_matrix)
                cost_list.append(cost_)
                
                items_emb[cur_pos] -= args.lr * x_grad_
                items_emb[contLS] -= 1.0/cont_len * args.lr * cont_grad_
                items_emb[nsample] -= args.lr * neg_grad_
                
            doc = Data.next()
        
        etime = time.time()
        print 'cost:',sum(cost_list)/len(cost_list)
        print 'cost time: ', etime-stime,'s'
        
        if (iter+1) % args.save_epochs == 0:
            print 'save model...'
            items_emb_dict = {}
            for key in Data.word_dic:
                idx = Data.get_index(key)
                items_emb_dict[key] = items_emb[idx]
            cPickle.dump(items_emb_dict, open(args.output+'_iter%d.pkl'%(iter+1), 'wb'))
    
        
    items_emb_dict = {}
    for key in Data.word_dic:
        idx = Data.get_index(key)
        items_emb_dict[key] = items_emb[idx]
    cPickle.dump(items_emb_dict, open(args.output+'.pkl', 'wb'))
    
    
    #----------------------------TEST--------------------------------#
    """
    word = u'china'
    idx = Data.get_index(word)
    vec = items_emb[idx].reshape(1,-1)
    simLS = []
    for i in range(len_vcob):
        if i == idx:
            continue
        veci = items_emb[i].reshape(1,-1)
        sim = cosine_similarity(veci,vec)
        simLS.append((i,sim))
    simLS.sort(key = lambda x : x[1],reverse = True)
    ResWDLs = []
    for it in simLS[:50]:
        idx = it[0]
        word = Data.get_item(idx)
        if type(word) == tuple:
            print word[0],word[1]
        else:
            print word
    """