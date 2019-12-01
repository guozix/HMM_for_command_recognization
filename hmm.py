import numpy as np

class HMM(object):

    
    def __init__(self, n_hidden, initial_prob=None, transition_prob=None):
        """保存隐含状态数目，初始化初始状态概率，状态转移概率

        Parameters:
        ----------
        initial_prob : (n_hidden,)
            状态初始概率

        transition_prob : (n_hidden, n_hidden)
            状态转移概率
        """
        self.n_hidden = n_hidden

        if initial_prob == None:
            initial_prob = np.ones((n_hidden))
            initial_prob /= n_hidden
        self.initial_prob = initial_prob

        if transition_prob == None:
            transition_prob = np.ones((n_hidden, n_hidden))
            transition_prob /= n_hidden
        self.transition_prob = transition_prob
    
    def log_likelihood(self, Q):
        '''似然与发射概率B有关, 而B可以连续可以离散, 故这里暂不定义'''
        raise NotImplementedError

    def maximize(self, Qs, epsilons, gammas):
        '''M部与发射概率B有关, 而B可以连续可以离散, 故这里暂不定义'''
        raise NotImplementedError

    
    def fit(self, Qs, iter_max = 10):
        """传入多组序列进行训练

        Parameters
        ----------
        Qs : list, each element Qs[i] in Qs is an (Ti * n_dim) array
            Qs为观测值序列组成的列表, 其中每个元素Q为一组观测向量序列/观测值序列, Q的行数为观测值的个数, 列数为每个观测值的维度
        
        iter_max : int, optional
            EM算法迭代的最大次数
        """
        # 将参数平铺开, 便于下面计算参数是否以某精度不变
        params = np.hstack((self.initial_prob.ravel(), self.transition_prob.ravel()))  
        
        # 其实这里应该也计算发射概率b的变化? d高斯时计算均值和协方差的变化 TODO:
        for _ in range(iter_max):              # 约定: 使用符号'_'表示不打算使用的变量, 作用: 可以防止出现未使用变量的警告 
            epsilons, gammas = self.expect(Qs)  # E部: 根据旧参数计算后验概率, 得到似然函数Q的系数, 也就得到了似然函数Q
            self.maximize(Qs, epsilons, gammas) # M部: 计算似然函数Q的极值点, 得到新的参数, 并更新到类里面

            params_new = np.hstack((self.initial_prob.ravel(), self.transition_prob.ravel()))
            # 逐元素 判断参数是否收敛, array中所有参数值收敛时返回True
            if np.allclose(params, params_new): 
                break
            else:
                params = params_new

        return 

    def expect(self, Qs):
        """E部, 计算epsilons, gammas

        Parameters
        ----------
        Qs : a list of some Q

        Returns
        -------
        epsilons : len(Qs), Tk , (n_hidden, n_hidden) 
            a list of some epsilon
        
        gammas : len(Qs), Tk , (n_hidden,)
            a list of some gamma

        Algorithms
        ----------
        E部 : 计算似然函数Q的系数: 根据旧参数计算后验概率 (注: M部根据不同的发射概率模型分别在子类实现)
            epsilon[k,t,i,j] = P(zt = i, zt+1 = j | Qk), 第k组观测值的条件下, t时刻 zt = i, zt+1 = j (状态对i,j)出现的概率
            gamma[k,t,i] = P(zt = i| Qk), 第k组观测值的条件下, t时刻 zt = i (状态i)出现的概率 
            使用前向变量alpha, 后向变量belta, 用DP简化计算
            alpha[t,i] = P(o1,...,ot, zt = i)
            belta[t,i] = P(ot+1,...,oT | zt=i)
            性质:
            1.  P(o1,...,oT) 
                = sum_i alpha(T, i)
            2.  P(o1,...,oT, zt=i) = P(o1,...,ot, zt=i) * P(ot+1,...,oT | zt=i, o1,...,ot) = P(o1,...,ot, zt=i) * P(ot+1,...,oT | zt=i)
                = alpha(t,i) * belta(t,i)
            3.  P(o1,...,oT) 
                = sum_i alpha(t,i) * belta(t,i)
            epsilon, gamma可用alpha, belta表示
            epsilon[k,t,i,j] = P(zt = i, zt+1 = j | Qk) = P(zt = i, zt+1 = j, Qk) / P(Qk) 
                            = alpha(t,i) * a(i,j) * b(t+1,j) * belta(t+1,j) / P(Qk)
                            = alpha(t,i) * a(i,j) * b(t+1,j) * belta(t+1,j) / sum_i alpha(Tk, i)
            gamma[k,t,i] = P(zt = i| Qk) 
                        = sum_j epsilon[k,t,i,j]
                        = alpha(t,i) belta(t,i) / P(Qk)
        关键:
            计算epsilon, gamma时, 先算log, 再用log法归一化, 最后再求指数, 得到最终结果.
        """
        # 忽略np.log(0)带来的警告
        with np.errstate(divide = 'ignore'):
            log_transition_prob = np.log(self.transition_prob)

        alphas, beltas = self.forward_and_backward(Qs)

        epsilons = list()
        gammas = list()
        # 对Qs中每个Q计算一次epsilon, gamma, 加入列表epsilons, gammas
        for k, Q in enumerate(Qs): 
            T = Q.shape[0]
            alpha = alphas[k]
            belta = beltas[k]
            
            log_likelihood = self.log_likelihood(Q)

            #计算epsilon
            #用log求发射概率,取指数再乘积后仍有可能得到0或者inf,故给发射概率取log, 而计算epsilon是还用乘积的话, 没有意义
            #所以epsilon用log计算, 再用log归一化, 最后再求指数
            epsilon = np.zeros((T, self.n_hidden, self.n_hidden), dtype = 'float64')      
            for t in range(T-1):
                for i in range(self.n_hidden):
                    for j in range(self.n_hidden):        
                        epsilon[t][i][j] = alpha[t][i] + log_transition_prob[i][j] + log_likelihood[t+1,j] + belta[t+1,j] 
                # log法归一化
                epsilon[t] = self.log_normalize(epsilon[t].reshape(self.n_hidden ** 2,)).reshape(self.n_hidden, self.n_hidden)

            with np.errstate(under = 'ignore'):
                epsilon = np.exp(epsilon)
            epsilons.append(epsilon)

            #计算gamma
            gamma = np.zeros((T, self.n_hidden), dtype = 'float64')
            for t in range(T):
                for i in range(self.n_hidden):
                    gamma[t][i] = alpha[t,i] + belta[t,i]
                # log法归一化
                gamma[t] = self.log_normalize(gamma[t])
            
            with np.errstate(under = 'ignore'):
                gamma = np.exp(gamma)
            gammas.append(gamma)

        return epsilons, gammas


    
    def forward_and_backward(self, Qs):
        """对数前后向算法

        Parameters
        ----------
        Qs : a list of some observe sequence Q

        Returns
        -------
        alphas : a list of some alpha

        beltas : a list of some belta

        Algorithms
        ----------
        前向算法
        input : Qs 同一HMM模型的多个观测序列
        output: alpha[k, t, i] 第k个序列alpha(t,i)的值
                alpha[k].shape == (Tk, n_hidden)
        DP algorithm:
        define: alpha(t,i) = P(o1,...,ot,zt=i)
        init  : alpha(1,i) = P(o1,z1=i) 
                            = P(z1=i)P(o1|z1=i)
                alpha(1,) = init_prob * b_1
        more  : alpha(t+1,i) = P(o1,...,ot+1,zt+1=i) 
                            = sum_j P(o1,...,ot+1,zt=j, zt+1=i) 
                            = sum_j P(o1,...,ot,zt=j) P(ot+1,zt+1=i|zt=j,o1,...,ot)
                            = sum_j alpha(t, j) P(ot+1,zt+1=i|zt=j)
                            = sum_j alpha(t, j) P(zt+1=i|zt=j)P(ot+1|zt+1=i,zt=j)
                alpha(t+1,i) = ( sum_j alpha(t, j) a[j,i] ) b[t+1,i]

        后向算法
        input : Qs 同一HMM模型的多个观测序列
        output: belta[k, t, i] belta(t,i)的值, t时刻状态i生成该时刻观测值的概率
                belta[k].shape == (Tk, n_hidden)
        DP algorithm:
        define: belta(t, i) = P(ot+1,...,oT|zt=i)
        init  : belta(T, i) = 1  :make sure P(o1,...,oT,zT=i) = alpha(T,i)*belta(T,i)
        more  : belta(t, i) = P(ot+1,...,oT|zt=i)
                            = sum_j P(zt+1=j,ot+1,ot+2,...,oT|zt=i)
                            = sum_j P(zt+1=j,ot+1|zt=i) P(ot+2,...,oT|zt+1=j,ot+1,zt=i)
                            = sum_j P(zt+1=j|zt=i) P(ot+1|zt+1=j,zt=i) P(ot+2,...,oT|zt+1=j)
                belta(t, i) = sum_j a(i,j) (b(t+1,j) belta(t+1,j))
                belta(t, i) = a(i,) X ( b(t+1,) * belta(t+1,) ) = a number
                belta(t, )  = [a(,)  X ( b(t+1,) * belta(t+1,) )] = an array, so don't need traverse
                
        """
        alphas = list()
        beltas = list()
        with np.errstate(divide = 'ignore'):
            log_initial_prob = np.log(self.initial_prob)
            log_transition_prob = np.log(self.transition_prob)
        for Q in Qs:
            T = Q.shape[0]   
            # 各时刻下各状态对该时刻观察值的对数发射概率
            log_likelihood = self.log_likelihood(Q) 

            # 计算alpha
            alpha = np.zeros((T, self.n_hidden))
            # 初始时刻的alpha
            for i in range(self.n_hidden):
                alpha[0,i] = log_initial_prob[i] + log_likelihood[0][i]
            for t in range(1, T):
                for i in range(self.n_hidden):
                    tmp = np.zeros(self.n_hidden,)
                    for j in range(self.n_hidden):
                        tmp[j] = alpha[t-1][j] + log_transition_prob[j][i]
                    alpha[t][i] = self.logsumexp(tmp) + log_likelihood[t][i]
            alphas.append(alpha)

            # 计算belta
            belta = list()
            # 初始时刻的belta
            belta_T = np.ones(self.n_hidden) 
            # python list method: list.insert(index, obj) 在指定位置插入元素
            belta.insert(0, belta_T)    
            # 这里及上文 t从T开始直到1, 故t时刻的似然存储在likelihood[t-1]内
            for t in range(T-2, -1, -1): 
                belta_t = np.zeros((self.n_hidden))
                for i in range(self.n_hidden):
                    tmp = np.zeros(self.n_hidden,)
                    for j in range(self.n_hidden):
                        tmp[j] = log_transition_prob[i][j] + log_likelihood[t+1,j] + belta[0][j]
                    belta_t[i] = self.logsumexp(tmp)
                belta.insert(0, belta_t) 
            belta = np.asarray(belta)
            beltas.append(belta)

        return alphas, beltas

    def generate_prob(self, Q):
        """计算Q似然, 亦即该hmm生成观测值序列Q的对数概率(密度)

        Parameters
        ----------
        Q : 2-d array, (T, n_dim)
            一组观测值序列

        Returns
        -------
        logP(Q|lambda) : float64
            对数似然概率
            原本公式是, P(Q|lambda) = sum_i alpha[T-1][i], 由于所得alpha是对数alpha, 故公式变为
            logP(Q|lambda) = log{sum_i exp(alpha[T-1][i])}
        """
        alphas, _ = self.forward_and_backward([Q])
        alpha = alphas[0]
        return self.logsumexp(alpha[len(Q)-1])

    def logsumexp(self, X):
        """compute logsumexp of array X, return log(expX[0] + expX[1] + ... + expX[n-1])

        Parameters
        ----------
        X : 1-d array
        
        Returns
        -------
        log(expX[0] + expX[1] + ... + expX[n-1])

        Algorithms
        ----------
        log(expX[0] + expX[1] + ... + expX[n-1]) = X_max + log(exp(X[0]-X_max) + exp(X[1]-X_max) + ... + exp(X[n-1]-X_max))
        这样exp的指数都会小于0, 
        如果X_max < 0, 那么指数会变大, 即X[i]-X_max > X[i], 可以防止下溢, TODO:上溢呢？
        如果X_max > 0, 那么指数会变小, 即X[i]-X_max < X[i], 可以防止上溢, TODO:下溢呢？

        Notes
        -----
        from hmmlearn https://github.com/hmmlearn/hmmlearn/blob/master/lib/hmmlearn/_hmmc.pyx#L26
        """
        X_max = np.max(X)
        if np.isinf(X_max):
            return -1 * np.inf
        acc = 0
        for i in range(X.shape[0]):
            acc += np.exp(X[i] - X_max)
        return np.log(acc) + X_max

    def log_normalize(self, a, axis = None):
        """Normalizes the input array so that the exponent of the sum is 1. 对数归一化

        Parameters
        ----------
        a : array
            Non-normalized input data.

        axis : int
            Dimension along which normalization is performed.

        Notes
        -----
        from hmmlearn https://github.com/hmmlearn/hmmlearn/blob/b643a9c7b8eaa652e4ad733efb96f0af4e956137/lib/hmmlearn/utils.py#L31
        """
        with np.errstate(under="ignore"):
            a_lse = self.logsumexp(a)
        a -= a_lse
        return a

    '''
    def forward(self, Qs):
        """
        前向算法
        input : Qs 同一HMM模型的多个观测序列
        output: alpha[k, t, i] 第k个序列alpha(t,i)的值
                alpha[k].shape == (Tk, n_hidden)
        DP algorithm:
        define: alpha(t,i) = P(o1,...,ot,zt=i)
        init  : alpha(1,i) = P(o1,z1=i) 
                            = P(z1=i)P(o1|z1=i)
                alpha(1,) = init_prob * b_1
        more  : alpha(t+1,i) = P(o1,...,ot+1,zt+1=i) 
                            = sum_j P(o1,...,ot+1,zt=j, zt+1=i) 
                            = sum_j P(o1,...,ot,zt=j) P(ot+1,zt+1=i|zt=j,o1,...,ot)
                            = sum_j alpha(t, j) P(ot+1,zt+1=i|zt=j)
                            = sum_j alpha(t, j) P(zt+1=i|zt=j)P(ot+1|zt+1=i,zt=j)
                alpha(t+1,i) = ( sum_j alpha(t, j) a[j,i] ) b[t+1,i]
                alpha(t+1,i) = alpha(t,) X a[,i] * b[t+1,i]
                alpha(t+1,)  = alpha(t,) X a[,]  * b[t+1,]
        """        
        alphas = list()
        for Q in Qs:
            T = Q.shape[0]   # 注意: shape, 不需要加括号
            log_likelihood = self.log_likelihood(Q)  # 求出各时刻各状态发射观察值的概率
            likelihood = np.exp(log_likelihood)
            alpha = list()
            alpha_1 = self.initial_prob * likelihood[0] # 初始时刻的alpha
            alpha.append(alpha_1)
            for t in range(1, T):
                alpha_t = np.matmul(alpha[-1], self.transition_prob) * likelihood[t]
                alpha.append(alpha_t)
            alpha = np.asarray(alpha)
            alphas.append(alpha)
            
        return alphas

    def backward(self, Qs):
        """
        后向算法
        input : Qs 同一HMM模型的多个观测序列
        output: belta[k, t, i] belta(t,i)的值, t时刻状态i生成该时刻观测值的概率
                belta[k].shape == (Tk, n_hidden)
        DP algorithm:
        define: belta(t, i) = P(ot+1,...,oT|zt=i)
        init  : belta(T, i) = 1  :make sure P(o1,...,oT,zT=i) = alpha(T,i)*belta(T,i)
        more  : belta(t, i) = P(ot+1,...,oT|zt=i)
                            = sum_j P(zt+1=j,ot+1,ot+2,...,oT|zt=i)
                            = sum_j P(zt+1=j,ot+1|zt=i) P(ot+2,...,oT|zt+1=j,ot+1,zt=i)
                            = sum_j P(zt+1=j|zt=i) P(ot+1|zt+1=j,zt=i) P(ot+2,...,oT|zt+1=j)
                belta(t, i) = sum_j a(i,j) (b(t+1,j) belta(t+1,j))
                belta(t, i) = a(i,) X ( b(t+1,) * belta(t+1,) ) = a number
                belta(t, )  = [a(,)  X ( b(t+1,) * belta(t+1,) )] = an array, so don't need traverse
        """
        beltas = list()
        for Q in Qs:
            T = Q.shape[0]
            log_likelihood = self.log_likelihood(Q)  # 求出各时刻各状态发射观察值的概率
            likelihood = np.exp(log_likelihood)
            belta = list()
            belta_T = np.ones(self.n_hidden) # 初始时刻的belta
            belta.insert(0, belta_T)
            for t in range(T-2, -1, -1): # 这里及上文 t从T开始直到1, 故t时刻的似然存储在likelihood[t-1]内
                belta_t = np.matmul(self.transition_prob, likelihood[t+1] * belta[0])
                belta.insert(0, belta_t) # python list method: list.insert(index, obj) 在指定位置插入元素
            belta = np.asarray(belta)
            
            for t in range(1,T-1):
                for i in range(self.n_hidden):
                    # belta(t, i) = sum_j a(i,j) (b(t+1,j) belta(t+1,j))
                    x = 0
                    for j in range(self.n_hidden):
                        x += self.transition_prob[i][j] * likelihood[t+1][j] * belta[t+1][j]
                    if np.abs(belta[t,i] - x) > 1e-3:
                        return False   
            beltas.append(belta)

        return beltas

    '''
    
    def viterbi(self, Q):
        # input: Q
        # output: 最佳状态链. 给定观测值Q时,该hmm模型中出现概率最大的状态链，从0开始计数状态索引
        T = Q.shape[0] #时间序列长度
        delta = np.full((T, self.n_hidden), -1*np.inf, dtype = 'float64') # Q.shape = (T, n_dim)
        pre =np.zeros((T, self.n_hidden), dtype = 'int64')
        log_likelihood = self.log_likelihood(Q)
        '''
        for i in range(self.initial_prob.shape[0]):
            self.initial_prob[i] += 1e-50
        for i in range(self.transition_prob.shape[0]):
            for j in range(self.transition_prob.shape[1]):
                self.transition_prob[i][j] += 1e-50  
        '''
        #初始化
        for i in range(self.n_hidden):
            delta[0][i] = np.log(self.initial_prob[i]) + log_likelihood[0][i]
            pre[0][i] = 0
        #
        for t in range(1, T):
            for j in range(self.n_hidden):
                for i in range(self.n_hidden):
                    if delta[t][j] < delta[t-1][i] + np.log(self.transition_prob[i][j]):
                        delta[t][j] = delta[t-1][i] + np.log(self.transition_prob[i][j])
                        pre[t][j] = i
                delta[t][j] += log_likelihood[t][j]
        maxDelta = -1 * np.inf
        q = -1
        for i in range(self.n_hidden):
            if delta[T-1][i] > maxDelta:
                maxDelta = delta[T-1][i]
                q = i
        states = [q]
        for t in range(T-1, 0, -1):
            q = pre[t][q]
            states.insert(0, q)
        return states

    

