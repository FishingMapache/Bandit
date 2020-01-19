import numpy as np
import cvxpy
import cvxopt
import random

def Sampling (d, K, J):
    # Sample theta_j. theta_J is a (J by d) matrix
    theta_J = np.random.multivariate_normal([0] * d, np.identity(d), J)

    # Sample x_k. X_K is a (d by K) matrix
    X_K = np.random.multivariate_normal([0] * d, np.identity(d), K).T

    return theta_J, X_K


def Reward_Generator(theta_J, X_K, K, J, n):
    # Declare a (K by J) matrix R_KJ
    R_KJ = np.zeros((K, J))

    # Declare an empty matrix R_KT with K rows
    R_KT = np.array([]).reshape((K, 0))

    for one_pass in range(n):
        for k in range(K):
            for j in range(J):
                reward = np.random.normal(np.dot(theta_J[j, :], X_K[:, k]), 1)
                R_KJ[k, j] = reward

        R_KT = np.concatenate((R_KT, R_KJ), axis=1)

    Expected_R_KJ = np.dot(theta_J, X_K).T

    return R_KT, Expected_R_KJ


def Integer_Programming(R_KT, B):
    # Flatten the true reward matrix
    Flatted_R_KT = R_KT.flatten()

    # Create a selection vector for all arm-context pairs
    selection = cvxpy.Variable(len(Flatted_R_KT), boolean=True)

    # Uni-cost system
    cost_vec = np.ones(len(Flatted_R_KT))

    # Total cost is less than B
    budget_constraint = cost_vec * selection <= B

    # Sum out the total reward
    total_reward = Flatted_R_KT * selection

    # Create the problem
    IP_problem = cvxpy.Problem(cvxpy.Maximize(total_reward), [budget_constraint])

    # Solving the problem
    IP_problem.solve(solver=cvxpy.GLPK_MI)

    total_reward = np.dot(Flatted_R_KT, selection.value)

    return total_reward


def Top_B_Columns(R_KT, T, B):
    list = []

    for t in range(T):
        max_reward = np.amax(R_KT[:, t])            # find the maximum reward for each time-step
        list.append(max_reward)

    from heapq import nlargest
    total_reward = sum(nlargest(B, list))

    return total_reward


def Linear_Programming(pi_j, avg_remaining_budget, J, mu_star):
    print("mu_star", mu_star)
    print("avg_remaining_budget", avg_remaining_budget)

    # Define and solve the LP problem.
    J_vec = cvxpy.Variable(J)

    objective = cvxpy.Maximize(cvxpy.sum(cvxpy.multiply(J_vec, mu_star)*pi_j))
    constraint = [cvxpy.sum(J_vec*pi_j) <= avg_remaining_budget]

    LP_problem = cvxpy.Problem(objective, constraint)

    LP_problem.solve()

    print("J_vec: ", J_vec.value)

    return J_vec.value


def Biased_Coin_Tossing(p):
    rand = random.random()

    if rand < p:
        action = 'Take Action'

    else:
        action = 'Skip'

    return action


def ALP(R_KT, Expected_R_KJ, B, J, T):
    pi_j = 1/float(J)       # probability that each user is sampled
    b_tau = B               # remaining_budget

    # Suppose the expected reward for all item-user pairs are known beforehand
    mu_star = []
    k_star = []

    for j in range(J):                                        # find the maximum expected reward of each user
        j_vec = Expected_R_KJ[:, j]
        max_reward = np.amax(j_vec)
        mu_star.append(max_reward)
        k_star.append(j_vec.argmax())

    mu_star = np.asarray(mu_star)

    total_reward = 0

    for n in range(T/J):
        for j in range(J):

            if b_tau == 0:
                return total_reward

            else:
                tau = T - n * J - j         # remaining time
                avg_remaining_budget = float(b_tau) / tau

                probs_of_action = Linear_Programming(pi_j, avg_remaining_budget, J, mu_star)
                #probs_of_action = random.uniform(0, 1)
                decision = Biased_Coin_Tossing(probs_of_action[j])             # Bernoulli Experiment
                print(decision)
                if decision == 'Take Action':
                    b_tau -= 1

                    true_reward = R_KT[k_star[j], (n * J + j)]         # reveal the reward
                    total_reward += true_reward

    return total_reward


def UCB_ALP(R_KT, T, B, K, J):
    b_tau = B                           # Initialize remaining budget as B
    pi_j = 1/float(J)                   # probability that each user is sampled

    # Initialize the no. of times an item-user pair is pulled up to t as 0
    C = np.zeros((K, J))

    # Initialize the average observed reward of all item-user pairs at time t as 0
    mu_bar = np.zeros((K, J))

    # Initialize the estimated expected reward of all item-user pairs at time t as 1
    mu_hat = np.ones((K, J))           # estimated upper bounds for item-user pairs at t

    # Create a vector for the highest estimated expected reward at t of each user
    mu_star = np.ones(J)

    total_reward = 0

    for t in range(T):

        if b_tau == 0:
            return total_reward

        else:
            tau = T - t                 # remaining time
            avg_remaining_budget = float(b_tau) / tau

            # Calculate which user it is
            j = int(t % J)
            print("j:", j)

            best_arm = np.argmax(mu_hat[:, j])                      # break ties arbitrarily for arms with highest estimated reward??

            probs_of_action = Linear_Programming(pi_j, avg_remaining_budget, J, mu_star)
            decision = Biased_Coin_Tossing(probs_of_action[j])      # Bernoulli Experiment
            print(decision)

            if decision == 'Take Action':
                b_tau -= 1

                true_reward = R_KT[best_arm, t]                     # reveal the reward
                print("true_reward: ", true_reward)

                total_reward += true_reward

                # Add one to the time of this item-user pair being pulled
                C[best_arm, j] += 1

                # Update the average observed reward of this item-user pair
                mu_bar[best_arm, j] = (mu_bar[best_arm, j] + true_reward) / C[best_arm, j]

                # Update the estimated expected reward (upper bound) of this item-user pair
                mu_hat[best_arm, j] = mu_bar[best_arm, j] + np.sqrt(np.log(t + 1)) / 2 * C[best_arm, j]

                # Update the best arm for this user
                best_arm = np.argmax(mu_hat[:, j])

                # Update the highest estimated expected reward for this user
                mu_star[j] = mu_hat[best_arm, j]

            print("mu_hat", mu_hat)

    return total_reward


def LinUCB_ALP(X_K, R_KT, T, B, K, J, d):
    b_tau = B                           # Initialize remaining budget as B
    pi_j = 1/float(J)                          # probability that each user is sampled

    # Create a (K by J) matrix for the estimated expected reward of all item-user pairs at time t
    mu_hat = np.ones((K, J))           # estimated upper bounds for item-user pairs at t

    # Create a vector for the highest estimated expected reward at t of each user
    mu_star = np.ones(J)

    # Initialize theta_J (d by J) for user features
    theta_J = np.zeros((d, J))

    # Initialize 3-d tensor A (J * d * d)
    A = np.zeros((J, d, d))

    # Initialize b (d by J)
    b = np.zeros((d, J))

    alpha = 1 + np.sqrt(np.log(2 / 0.95) / 2)           # what should be alpha, 1 / t?

    for j in range(J):
        A[j:, :, :] = np.identity(d)
        theta_J[:, j] = np.dot(np.linalg.inv(A[j, :, :]), b[:, j])

        for k in range(K):
            mu_hat[k, j] = np.dot(theta_J[:, j].T, X_K[:, k]) + alpha * np.sqrt(
                np.dot(np.dot(X_K[:, k].T, np.linalg.inv(A[j, :, :])), X_K[:, k]))

        mu_star[j] = np.amax(mu_hat[:, j])

    total_reward = 0

    for t in range(T):

        if b_tau == 0:
            return total_reward

        else:
            tau = T - t                 # remaining time
            avg_remaining_budget = float(b_tau) / tau

            # Calculate which user it is
            j = int(t % J)
            print("J", j)

            best_arm = np.argmax(mu_hat[:, j])

            probs_of_action = Linear_Programming(pi_j, avg_remaining_budget, J, mu_star)
            # probs_of_action = random.uniform(0, 1)

            decision = Biased_Coin_Tossing(probs_of_action[j])       # Bernoulli Experiment
            print(decision)

            if decision == 'Take Action':
                b_tau -= 1

                true_reward = R_KT[best_arm, t]                     # reveal the reward
                total_reward += true_reward

                A[j:, :, :] += np.dot(X_K[:, best_arm],  X_K[:, best_arm].T)
                b[:, j] += true_reward * X_K[:, best_arm]

                theta_J[:, j] = np.dot(np.linalg.inv(A[j, :, :]), b[:, j])

                for k in range(K):
                    mu_hat[k, j] = np.dot(theta_J[:, j].T, X_K[:, k]) + alpha * np.sqrt(
                        np.dot(np.dot(X_K[:, k].T, np.linalg.inv(A[j, :, :])), X_K[:, k]))
                print(mu_hat)
                best_arm = np.argmax(mu_hat[:, j])
                mu_star[j] = mu_hat[best_arm, j]

    return total_reward


def main():
    d = 3                # number of features
    K = 5                # number of arms
    J = 4                # number of users
    n = 20               # time that each user appear
    T = J * n            # total number of time-steps
    B = int(0.7 * T)     # budget

    theta_J, X_K = Sampling(d, K, J)

    # Generate the true reward for all arn-context pairs
    R_KT, Expected_R_KJ = Reward_Generator(theta_J, X_K, K, J, n)

    reward_IP = Integer_Programming(R_KT, B)
    print("----------------------------------------------------------------------------------------")
    print("The total reward obtained by the IP experiment is: ", reward_IP)
    print("----------------------------------------------------------------------------------------")

    reward_top_B_cols = Top_B_Columns(R_KT, T, B)
    print("The total reward obtained by pulling one arm in the best B time-steps is: ", reward_top_B_cols)
    print("----------------------------------------------------------------------------------------")

    reward_ALP = ALP(R_KT, Expected_R_KJ, B, J, T)
    print("The total reward obtained by the ALP experiment is: ", reward_ALP)
    print("The regret compared with the IP experiment is:  ", reward_IP - reward_ALP)
    print("The regret compared with the Top_B_Columns experiment is:  ", reward_top_B_cols - reward_ALP)
    print("----------------------------------------------------------------------------------------")
    '''
    reward_UCB_ALP = UCB_ALP(R_KT, T, B, K, J)
    print("The total reward obtained by the UCB-ALP experiment is: ", reward_UCB_ALP)
    print("The regret compared with the IP experiment is: ", reward_IP - reward_UCB_ALP)
    print("The regret compared with the Top_B_Columns experiment is: ", reward_top_B_cols - reward_UCB_ALP)
    print("The regret compared with the ALP experiment is: ", reward_ALP - reward_UCB_ALP)
    print("----------------------------------------------------------------------------------------")

    reward_LinUCB_ALP = LinUCB_ALP(X_K, R_KT, T, B, K, J, d)
    print("The total reward obtained by the LinUCB-ALP experiment is: ", reward_LinUCB_ALP)
    print("The regret compared with the IP experiment is: ", reward_IP - reward_LinUCB_ALP)
    print("The regret compared with the Top_B_Columns experiment is: ", reward_top_B_cols - reward_LinUCB_ALP)
    print("The regret compared with the ALP experiment is: ", reward_ALP - reward_LinUCB_ALP)
    '''

if __name__ == "__main__":
    main()