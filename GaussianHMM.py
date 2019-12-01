from hmm import HMM
from MultivariateGaussian import MultivariateGaussian
import numpy as np
import sklearn.cluster as cluster
class GaussianHMM(HMM):

    def __init__(self, n_hidden, n_dim, initial_prob=None, transition_prob=None):
        # (super)保存隐含状态数目，初始化初始状态概率，状态转移概率
        # (GaussianHMM) 初始化观测值维度，每个状态发射概率为高斯分布——初始化均值和方差
        super().__init__(n_hidden, initial_prob, transition_prob)
        self.n_dim = n_dim
        self.min_cov = 1e-3
    
    def kmeans_init(self, Qs):
        # 调用kmeans初始化
        X = np.vstack(tuple(Qs)) # 将各个Q按垂直方向排列, 每个Q是一个Tk*n_dim矩阵, 最终得到(sum_k Tk) * n_dim 矩阵
        kmeans = cluster.KMeans(n_clusters=self.n_hidden,
                                    random_state=None)
        kmeans.fit(X)

        self.means = kmeans.cluster_centers_
        #cv = np.cov(X.T) + self.min_cov * np.eye(X.shape[1])
        #self.covs = np.tile(np.diag(cv), (self.n_hidden, 1))
        self.covs = np.tile(np.ones(X.shape[1]), (self.n_hidden, 1))
        

    def viterbi_init(self, Qs, iter_max=1):
        '''
        viterbi初始化，输入用于训练的数据
        kmeans解决了发射均值的初始化，经过kmeans之后，初始化概率是平均值、转移矩阵是平均值、发射概率cov统一是所有样本的方差
        viterbi将要初始化初始概率、转移矩阵A、均值：
        
        对于k个训练样本，使用viterbi解码得到k条状态序列，每条长度为Tk，可以将每个mfcc向量对应到一个状态，状态为012345
        初始化概率：为保证不会有0概率，对于k条状态序列中没有作为开头出现的状态处理：0.1/(k+0.1*n_hidden),出现过n次的:(0.1+n)/(k+0.1*n_hidden),0.1设置为平滑因子factor
        转移矩阵：使用二维矩阵trans统计k条状态序列中状态转移的情况，trans[i][j]表示由i到j的次数，同样，对于可能出现0概率的情况用平滑因子处理了
        均值，统计属于各个状态的所有mfcc向量的平均向量
        '''
        Qs = np.array(Qs)
        for iter in range(iter_max):
            #先求状态序列
            state_sequences = []
            for Q in Qs:
                state_sequences.append(self.viterbi(Q))
            state_sequences = np.array(state_sequences)
            num_of_seq = len(state_sequences)
            
            #计算初始概率
            state_count = np.zeros(self.n_hidden)
            smooth_factor1 = 1e-5
            for i in range(num_of_seq):
                state_count[state_sequences[i][0]] += 1
            start_prob = []
            for i in range(self.n_hidden):
                prob = (state_count[i]+smooth_factor1)/(num_of_seq+smooth_factor1*self.n_hidden)
                start_prob.append(prob)
            self.initial_prob = np.array(start_prob)
            
            #初始化转移矩阵
            smooth_factor2 = 1e-5
            trans_count = np.zeros((self.n_hidden,self.n_hidden))
            for i in range(num_of_seq):
                seq_length = len(state_sequences[i])
                for j in range(1,seq_length):
                    trans_count[state_sequences[i][j-1]][state_sequences[i][j]] += 1
            trans_count += smooth_factor2
            sum_of_line = trans_count.sum(axis = 1)
            trans_prob = (trans_count.T/sum_of_line).T
            self.transition_prob = trans_prob
            
            #计算均值
            state_count = np.zeros(self.n_hidden)
            sum_of_vecs = np.zeros((self.n_hidden,self.n_dim))
            for seq in range(num_of_seq):
                for i in range(len(state_sequences[seq])):
                    state_count[state_sequences[seq][i]] += 1
                    #print("----------",Qs.shape)
                    sum_of_vecs[state_sequences[seq][i]] += Qs[seq][i]
            average_of_vecs = (sum_of_vecs.T/state_count).T
            self.means = average_of_vecs
            

    def log_likelihood(self, X):
        """计算X中各观测值的似然, 每时刻每种状态生成该时刻观测值的概率
        Gaussian_hmm中, 由于发射概率模型是高斯分布, 只给定了分布的参数, 而发射概率密度B未显式给定, 下面根据连续分布函数求出离散的B.
        命名原因: '似然'的含义：计算生成数据的概率; 此处的'似然'的含义: 假设发射概率为多维高斯分布，计算每时刻每种状态生成给定观测值的概率
        
        Paramaters:
        ----------
        X : 2-dim array, shape (T, n_dim)
           X为一组观测值(向量)序列, X[t]为t时刻的观测值向量
        
        Returns
        -------
        log_likelihood : 2-dim array, shape (T, n_hidden)
            log_likelihood = logB
            log_likelihood[t,i] = logB[t,i] = logP(ot|zt = i) = t时刻状态i生成该时刻观测值X[t]的对数概率
             
        """
        T = X.shape[0]
        '''
        diff = np.zeros((T, self.n_hidden, self.n_dim), dtype = 'float64')   
        # 矩阵求法: diff = X[:,None,:] - self.means  给定时刻t， 计算ot与每个状态发射均值的差值，所以要先扩展X，使之在时刻t有n_hidden个相同的值，再分别减去各状态的发射均值
        for t in range(T):      
            for i in range(self.n_hidden):
                diff[t][i] = X[t] - self.means[i]   
        '''
        log_likelihood = np.zeros((T, self.n_hidden), dtype = 'float64')
        for t in range(T):
            for i in range(self.n_hidden):
                # 根据状态i的正态分布, 计算t时刻观测值Q[t]的对数发射概率
                log_pb = MultivariateGaussian(self.means[i], self.covs[i]).log_prob(X[t])  # t时刻, 第i个状态的高斯分布生成该时刻观测值X[t]的概率
                log_likelihood[t][i] = log_pb

        return log_likelihood
    
    def maximize(self, Qs, epsilons, gammas):
        """
        M部, 根据epsilons、gammas计算gaussian_hmm参数
        
        Parameters:
        ----------
        Qs       (num, T_k)
        epsilons (num, T_k, n_hidden, n_hidden)
        gammas   (num, T_k, n_hidden)
        
        update:
        -------
        initial_prob[i] = gamma(1,i)
                shape  = (n_hidden, )
        transition_prob[i,j] = sum_t epsilon_t(i, j) / sum_t gamma_t(i)
                shape  = (n_hidden, n_hidden)
        means[i] = ui = sum_t gamma(t, i) * ot / sum_t gamma(t, i)
                    = gamma(,i) X Q / sum_t gamma(t, i)
        """     
        # 观测序列的个数
        num = len(Qs) 

        # 估计初始概率
        self.initial_prob = np.zeros(self.n_hidden, dtype = 'float64')
        for k in range(num):
            self.initial_prob += gammas[k][0]
        self.initial_prob /= num      

        # 估计转移概率
        transition_prob = np.zeros((self.n_hidden, self.n_hidden))
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                for epsilon in epsilons:
                    transition_prob[i][j] += epsilon[:,i,j].sum()
        for i in range(self.n_hidden):
            tmp = 0
            for k in range(num):
                tmp += gammas[k][:,i].sum()
            transition_prob[i] /= tmp
        self.transition_prob = transition_prob

        # 估计均值
        for i in range(self.n_hidden):
            ui = np.zeros((self.n_dim,))
            # denom计算gamma[:,i]之和, denom译为'分母项'
            denom = 0 
            for k in range(num):
                gamma = gammas[k]
                for t in range(Qs[k].shape[0]):
                    ui += gamma[t][i] * Qs[k][t]
                    denom += gamma[t][i]
            ui /= denom
            self.means[i] = ui

        # 估计协方差, 并将协方差矩阵转成对角阵, 再转成1*n_dim的数组
        covs = np.zeros((self.n_hidden, self.n_dim, self.n_dim), dtype = 'float64')
        denom = np.zeros(self.n_hidden)
        for k in range(num):
            Q = Qs[k]
            T = Q.shape[0]
            gamma = gammas[k]
            for t in range(T):
                for i in range(self.n_hidden):
                    diff = Q[t] - self.means[i]
                    diff.shape = (1, diff.shape[0])
                    covs[i] += gamma[t][i] * np.matmul(diff.T, diff)
                    denom[i] += gamma[t][i]               
        #print(sum([gamma.sum(axis = 0) for gamma in gammas]).shape): (n_hidden,) 没问题
        for i in range(self.n_hidden):
            covs[i] /= denom[i]
            # 保证最小方差
            covs[i] += 1e-3 # TODO: 应该遍历对角每个元素, covs[k][i][i] += np.sign(covs[k][i][i]) * 1e-3

        new_covs = np.zeros((self.n_hidden, self.n_dim))
        for i in range(self.n_hidden):
            new_covs[i] = np.diag(covs[i])
        self.covs = new_covs
        
        return

    '''
    def supervision_train(self, Q, states):
        #states = self.viterbi(Q)
        cnt = np.zeros(self.n_hidden, dtype = 'int64')

        # 初始化 means
        means = np.zeros((self.n_hidden, self.n_dim), dtype = 'float64')
        for t, state in enumerate(states):
            cnt[state] += 1
            means[state] += Q[t]
        for state in states:
            means[state] /= cnt[state]

        # 初始化 covs
        covs = np.zeros((self.n_hidden, self.n_dim, self.n_dim), dtype = 'float64')
        for i in range(self.n_hidden):
            covs[i] = np.eye(self.n_dim, self.n_dim)
        for t, state in enumerate(states):
            diff = Q[t] - means[state]
            diff.shape = (1, self.n_dim)
            covs[state] += np.matmul(diff.T, diff)
        for state in states:
            covs[state] /= cnt[state]

        # 初始化 transition_prob
        trans_cnt = np.zeros((self.n_hidden, self.n_hidden), dtype = 'int64')
        for t, state in enumerate(states[:-1]):
            trans_cnt[state][states[t+1]] += 1
        transition_prob = np.zeros((self.n_hidden, self.n_hidden))
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                if np.abs(trans_cnt[i].sum()) > 1e-5:
                    transition_prob[i][j] = trans_cnt[i][j] / trans_cnt[i].sum()
        # 初始化 initial_prob
        initial_prob = np.zeros(self.n_hidden, dtype = 'float64')
        initial_prob[states[0]] = 1
        return initial_prob, transition_prob, means, covs
    def viterbi_init(self, Qs, iter_max = 5):
        params = np.hstack((self.initial_prob.ravel(), self.transition_prob.ravel(), self.means.ravel(), self.covs.ravel()))
        for _ in range(iter_max):   
            sum_initial_prob = np.zeros((self.n_hidden))
            sum_transition_prob = np.zeros((self.n_hidden, self.n_hidden))
            sum_means = np.zeros((self.n_hidden, self.n_dim))
            sum_covs = np.zeros((self.n_hidden, self.n_dim, self.n_dim))
            for Q in Qs:
                states = self.viterbi(Q)
                #print('states:',states)
                initial_prob, transition_prob, means, covs = self.supervision_train(Q, states)  # 给定观测值序列Q 和 状态序列states, 利用监督学习/频率=概率 估计hmm参数
                sum_initial_prob += initial_prob
                sum_transition_prob += transition_prob
                sum_means += means
                sum_covs += covs
            self.initial_prob = sum_initial_prob / len(Qs)
            self.transition_prob = sum_transition_prob / len(Qs)
            self.means = sum_means / len(Qs)
            
            sum_covs /= len(Qs)
            # 将协方差矩阵转成对角阵
            for k in range(self.n_hidden):
                for i in range(self.n_dim):
                    for j in range(self.n_dim):
                        if i == j and abs(sum_covs[k][i][j]) < 1e-5:
                            sum_covs[k][i][j] = np.sign(sum_covs[k][i][j]) * 1e-5
                        elif i != j:
                            sum_covs[k][i][j] = 0
            flag = 0
            for k in range(self.n_hidden):
                for i in range(self.n_dim):
                    for j in range(self.n_dim):    
                        if (i == j and sum_covs[k][i][j] == 0) or (i != j and sum_covs[k][i][j] != 0):
                            flag = 1
                            break
            #if flag:
            #    print('!!!!!!!!!!!!!')
            #    print(covs)
            #    return
            self.covs = sum_covs

            params_new = np.hstack((self.initial_prob.ravel(), self.transition_prob.ravel(), self.means.ravel(), self.covs.ravel()))
            if np.allclose(params, params_new): # 逐元素 判断参数是否收敛, array中所有参数值收敛时返回True
                break
            else:
                params = params_new
        return
    '''
