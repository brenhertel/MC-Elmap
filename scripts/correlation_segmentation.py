import numpy as np
import matplotlib.pyplot as plt

def cos_similarity(a, b):
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0
    return num / denom

class Correlation_Segmentation(object):

    def __init__(self, demo, sub_tasks, metric='SSE'):
        self.full_demo = demo
        (self.n_pts, self.n_dims) = np.shape(self.full_demo)
        self.sub_tasks = sub_tasks
        self.M = len(self.sub_tasks)
        self.sim_metric = metric
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y']
        
    def segment(self):
        self.Q = -float('inf') * np.ones((self.n_pts, self.M))
        for i in range(self.M):
            t_i = len(self.sub_tasks[i])
            for j in range(self.n_pts - t_i + 1):
                if self.sim_metric == 'CCS':
                    sim_ij = sum([np.dot(self.sub_tasks[i][m, :], self.full_demo[m+j, :]) for m in range(t_i)])
                elif self.sim_metric == 'SSE':
                    sim_ij = sum([-(np.linalg.norm(self.sub_tasks[i][m, :] - self.full_demo[m+j, :])**2) for m in range(t_i)])
                elif self.sim_metric == 'COS':
                    sim_ij = sum([cos_similarity(self.sub_tasks[i][m+1, :] - self.sub_tasks[i][m, :], self.full_demo[m+j+1, :] - self.full_demo[m+j, :]) for m in range(t_i-1)])
                else:
                    #print('Similarity not implemented!')
                    sim_ij = 0
                #print(sim_ij)
                self.Q[j:j+t_i, i] = np.maximum(self.Q[j:j+t_i, i], sim_ij)
        print(self.Q)
        self.Z = np.argmax(self.Q, axis=1)
        return self.Z
        
    def segment2(self):
        self.Q = -float('inf') * np.ones((self.n_pts, self.M))
        for i in range(self.M):
            t_i = len(self.sub_tasks[i])
            for j in range(self.n_pts - t_i + 1):
                if self.sim_metric == 'CCS':
                    sim_ij = sum([np.dot(self.sub_tasks[i][m, :], self.full_demo[m+j, :]) for m in range(t_i)])
                elif self.sim_metric == 'SSE':
                    sim_ij = sum([-(np.linalg.norm(self.sub_tasks[i][m, :] - self.full_demo[m+j, :])**2) for m in range(t_i)])
                elif self.sim_metric == 'COS':
                    sim_ij = sum([cos_similarity(self.sub_tasks[i][m+1, :] - self.sub_tasks[i][m, :], self.full_demo[m+j+1, :] - self.full_demo[m+j, :]) for m in range(t_i-1)])
                else:
                    #print('Similarity not implemented!')
                    sim_ij = 0
                #print(sim_ij)
                self.Q[j:j+t_i, i] = np.maximum(self.Q[j:j+t_i, i], sim_ij)
        print(self.Q)
        self.Z = -1*np.ones((self.n_pts, ))
        for i in range(self.M):
            start_idx = np.argmax(self.Q[:, i])
            t_i = len(self.sub_tasks[i])
            if start_idx + t_i < self.n_pts:
                self.Z[start_idx:start_idx+t_i] = i
            else:
                self.Z[start_idx:] = i
        return self.Z
        
    def segment3(self):
        self.Q = -float('inf') * np.ones((self.n_pts, self.M))
        for i in range(self.M):
            t_i = len(self.sub_tasks[i])
            for j in range(self.n_pts - t_i + 1):
                if self.sim_metric == 'CCS':
                    sim_ij = sum([np.dot(self.sub_tasks[i][m, :], self.full_demo[m+j, :]) for m in range(t_i)])
                elif self.sim_metric == 'SSE':
                    sim_ij = sum([-(np.linalg.norm(self.sub_tasks[i][m, :] - self.full_demo[m+j, :])**2) for m in range(t_i)])
                elif self.sim_metric == 'COS':
                    sim_ij = sum([cos_similarity(self.sub_tasks[i][m+1, :] - self.sub_tasks[i][m, :], self.full_demo[m+j+1, :] - self.full_demo[m+j, :]) for m in range(t_i-1)])
                else:
                    #print('Similarity not implemented!')
                    sim_ij = 0
                #print(sim_ij)
                self.Q[j:j+t_i, i] = np.maximum(self.Q[j:j+t_i, i], sim_ij)
        print(self.Q)
        self.Z = -1*np.ones((self.n_pts, ))
        self.inds = np.zeros((self.M, ))
        for i in range(self.M):
            for j in range(self.M):
                if np.max(self.Q) == np.max(self.Q[:, j]) and np.max(self.Q) > -float('inf'):
                    start_idx = np.argmax(self.Q[:, j])
                    t_i = len(self.sub_tasks[j])
                    next_ind = t_i
                    for i in range(t_i):
                        if start_idx+i >= self.n_pts or self.Z[start_idx+i] > -1:
                                next_ind = i
                                break
                    if start_idx + next_ind < self.n_pts:
                        self.Z[start_idx:start_idx+next_ind] = j
                        self.Q[start_idx:start_idx+next_ind, :] = -float('inf') * np.ones((next_ind, self.M))
                    else:
                        self.Z[start_idx:] = j
                        self.Q[start_idx:, :] = -float('inf') * np.ones((self.n_pts-start_idx, self.M))
                    self.inds[j] = start_idx
                    self.Q[:, j] = -float('inf') * np.ones((self.n_pts, ))
                    print(self.inds)
        return self.Z
        
    def plot(self):
        plt.figure()
        if self.n_dims == 1:
            plt.plot(self.full_demo, 'k')
            for i in range(self.n_pts):
                plt.plot(i, self.full_demo[i, 0], self.colors[self.Z[i]] + '.', ms=10)
        elif self.n_dims == 2:
            plt.plot(self.full_demo[:, 0], self.full_demo[:, 1], 'k')
            for i in range(self.n_pts):
                plt.plot(self.full_demo[i, 0], self.full_demo[i, 1], self.colors[self.Z[i]] + '.', ms=10)
        plt.show()
        
def main():
    n = 50
    t = np.linspace(0, np.pi, n).reshape((n, 1))
    
    x1 = 3*np.sin(2*t)
    x2 = 5*np.sin(t)*np.cos(t)

    x = np.vstack((x1, x2))
    
    CS = Correlation_Segmentation(x, [x1, x2], metric='SSE')
    CS.segment()
    CS.plot()
    
if __name__ == '__main__':
    main()