import numpy as np
from get_mfc_data import get_mfc_data
from GaussianHMM import GaussianHMM

if __name__ == "__main__":
    #datas = get_mfc_data('C:/Users/18341/Desktop/book/听觉/实验3-语音识别/语料/features/')
    datas = get_mfc_data('F:/HIT/大三上/视听觉/lab3/组/gzx_sound_mfcc/')
    
    # 每个类别创建一个hmm, 并用kmeans初始化hmm
    hmms = dict()
    for category in datas:
        Qs = datas[category]
        n_hidden = 5     
        n_dim = Qs[0].shape[1]

        hmm = GaussianHMM(n_hidden,n_dim)
        hmm.kmeans_init(Qs[:-3])
        hmm.viterbi_init(Qs) #
        hmms[category] = hmm
    #print(len(Qs))
    #print(len(Qs[0]))
    #print(len(Qs[1]))
    
    # 训练每个hmm
    print('start fit')
    for category in hmms:
        hmm = hmms[category]
        #print(hmm.covs)
        Qs = datas[category]
        hmm.fit(Qs[:-3], iter_max = 5)
        hmms[category] = hmm
        print(category, ':fit success')
    
    # 测试, 打印得分和最终正确率
    correct_num = 0
    for real_category in datas:
        for test_sample in datas[real_category][-3:]:
            print('real_category:', real_category)
            max_score = -1 * np.inf
            predict_category = -1
            for test_category in hmms:
                hmm = hmms[test_category]
                score = hmm.generate_prob(test_sample)
                print('test category ', test_category, '\'s score: ', score)
                if score > max_score:
                    max_score = score
                    predict_category = test_category
            if predict_category == real_category:
                correct_num += 1
            print('predict_category:',predict_category)
    print(correct_num / (3*5))
    
    #输出参数
    print("params:\n")
    print(hmms['1'].initial_prob)
    print(hmms['1'].transition_prob)
    print('-------------')
    print(hmms['2'].initial_prob)
    print(hmms['2'].transition_prob)
    print('-------------')
    print(hmms['3'].initial_prob)
    print(hmms['3'].transition_prob)
    print('-------------')
    print(hmms['4'].initial_prob)
    print(hmms['4'].transition_prob)
    print('-------------')
    print(hmms['5'].initial_prob)
    print(hmms['5'].transition_prob)
    
  