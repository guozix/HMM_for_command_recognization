import numpy as np
from hmmlearn import hmm
from get_mfc_data import get_mfc_data

if __name__ == "__main__":
    #datas = get_mfc_data('C:/Users/18341/Desktop/book/听觉/实验3-语音识别/语料/features/')
    datas = get_mfc_data('F:/HIT/大三上/视听觉/lab3/组/gzx_sound_mfcc/')
    
    hmms = dict()
    for category in datas:
        Qs = datas[category]
        n_hidden = 6
        model = hmm.GaussianHMM(n_components = 5, n_iter = 5, tol = 0.01, covariance_type="diag")
        vstack_Qs = np.vstack(tuple(Qs[:-3]))
        model.fit(vstack_Qs, [Q.shape[0] for Q in Qs[:-3]])
        print('success fit')
        hmms[category] = model
        
    #test
    correct_num = 0
    for category in datas:
        for test_sample in datas[category][-3:]:
            print('real_category:', category)
            max_score = -1 * np.inf
            predict = -1
            for predict_category in hmms:
                model = hmms[predict_category]
                score = model.score(test_sample)
                print('category', predict_category, '. score:', score)
                if score > max_score:
                    max_score = score
                    predict = predict_category
                    #print('predict_category', predict_category)
            if predict == category:
                correct_num += 1
            print('predict_category:',predict)
    print(correct_num / (3*5))