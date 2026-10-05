#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #panda is used to loas and manipulate data and for one-hot encoding
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pandas as pd


# In[2]:


data = pd.read_excel('/Users/ainalatifa/Desktop/data.xlsx')


# In[3]:


data.head()


# In[4]:


data = data.drop(columns=['TANGGAL'])


# In[5]:


data = data.drop(columns=['DDD_CAR'])


# In[6]:


data.head()


# In[7]:


data.replace("-", np.nan, inplace=True)


# In[8]:


data.head()


# In[9]:


data.info()


# In[10]:


display(data.isna().sum())


# In[11]:


data = data.dropna()


# In[12]:


display(data.isna().sum())


# In[13]:


data.describe()


# In[14]:


data.head()


# In[15]:


data.info()


# In[16]:


X_encoded = data.drop(columns=['RR'])
y = data['RR']


# In[17]:


print(X_encoded.head())


# In[18]:


print(X_encoded.head())


# In[19]:


print(y.head())


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[21]:


print(X_train.head())


# In[22]:


# Cek ukuran hasil split
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# In[23]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[24]:


scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(
    y_train.values.reshape(-1, 1)
)

y_test_scaled = scaler_y.transform(
    y_test.values.reshape(-1, 1)
)


# In[25]:


y_train_scaled = scaler_y.fit_transform(
    y_train.values.reshape(-1, 1)
).ravel()

y_test_scaled = scaler_y.transform(
    y_test.values.reshape(-1, 1)
).ravel()


# In[ ]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

# ==========================================================
# OBJECTIVE FUNCTION
# ==========================================================
def objective(params, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, epsilon=0.1):
    """Objective function for MOPSO-SVR:
       f1 = MSE (to minimize)
       f2 = number of support vectors (to minimize)
    """
    C, gamma = params
    model = SVR(C=C, gamma=gamma, kernel='rbf')
    model.fit(X_train_scaled, y_train_scaled)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test_scaled, y_pred)
    n_sv = len(model.support_)  # Support Vector cardinality

    return mse, n_sv


# ==========================================================
# DOMINATION CHECK
# ==========================================================
def dominates(a, b):
    """Check if solution a dominates b (for minimization problems)."""
    return np.all(a <= b) and np.any(a < b)


# ==========================================================
# MOPSO-SVR MAIN FUNCTION
# ==========================================================
def mopso_svr(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
              iter_max=100, swarm_size=40):
    bounds = {'C': (0.1, 10), 'gamma': (0.1, 10)}

    # Inisialisasi partikel dan kecepatan
    particles = np.array([
        [np.random.uniform(*bounds['C']), np.random.uniform(*bounds['gamma'])]
        for _ in range(swarm_size)
    ])
    velocities = np.zeros_like(particles)

    # Inisialisasi personal best
    pbest = particles.copy()
    pbest_f = np.array([
        objective(p, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
        for p in pbest
    ])

    # Inisialisasi archive (Pareto front)
    archive = pbest.copy()
    archive_f = pbest_f.copy()

    w, c1, c2 = 1.0, 2.0, 2.0

    # ======================================================
    # ITERASI MOPSO
    # ======================================================
    for it in range(iter_max):
        fitness = np.array([
            objective(p, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
            for p in particles
        ])

        # Update pbest
        for i in range(swarm_size):
            if dominates(fitness[i], pbest_f[i]):
                pbest[i], pbest_f[i] = particles[i], fitness[i]

        # Update archive (Pareto non-dominated solutions)
        all_particles = np.vstack((archive, particles))
        all_fitness = np.vstack((archive_f, fitness))
        non_dominated = []
        for i, f in enumerate(all_fitness):
            if not any(dominates(other, f) for j, other in enumerate(all_fitness) if j != i):
                non_dominated.append(i)
        archive = all_particles[non_dominated]
        archive_f = all_fitness[non_dominated]

        # Pilih global best acak dari archive
        gbest = archive[np.random.randint(len(archive))]

        # Update velocity & position
        for i in range(swarm_size):
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))
            particles[i] += velocities[i]

            # Bound check
            particles[i, 0] = np.clip(particles[i, 0], *bounds['C'])
            particles[i, 1] = np.clip(particles[i, 1], *bounds['gamma'])

    # ======================================================
    # HASIL AKHIR (AMBIL BERDASARKAN MSE TERKECIL)
    # ======================================================
    best_idx = np.argmin(archive_f[:, 0])
    best_params = archive[best_idx]
    best_mse = archive_f[best_idx][0]
    best_nsv = archive_f[best_idx][1]

    return best_params, best_mse, best_nsv


# ==========================================================
# MULTI-RUN EKSEKUSI
# =========================================================
n_runs = 10
mse_list, nsv_list, best_params_list = [], [], []

print("===== PROSES MOPSO-SVR (MSE & #SV) =====")
for run in range(n_runs):
    best_params, best_mse, best_nsv = mopso_svr(
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
        iter_max=100, swarm_size=40
    )
    mse_list.append(best_mse)
    nsv_list.append(best_nsv)
    best_params_list.append(best_params)

    print(f"Run {run+1:02d} | Best C={best_params[0]:.4f}, gamma={best_params[1]:.4f}, "
          f"MSE={best_mse:.5f}, #SV={int(best_nsv)}")


# In[26]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

# ==========================================================
# OBJECTIVE FUNCTION
# ==========================================================
def objective(params, X_train_scaled, y_train_scaled,
              X_test_scaled, y_test_scaled):

    C, gamma = params

    model = SVR(
        C=C,
        gamma=gamma,
        kernel='rbf'
    )

    model.fit(X_train_scaled, y_train_scaled)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test_scaled, y_pred)
    n_sv = len(model.support_)

    return mse, n_sv


# ==========================================================
# DOMINATION CHECK
# ==========================================================
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


# ==========================================================
# RANDOM SEARCH SVR
# ==========================================================
def random_search_svr(X_train_scaled, y_train_scaled,
                      X_test_scaled, y_test_scaled,
                      iter_max=100):

    bounds = {
        'C': (0.1, 10),
        'gamma': (0.1, 10)
    }

    solutions = []
    fitnesses = []

    # ==========================================
    # RANDOM SEARCH
    # ==========================================
    for _ in range(iter_max):

        C = np.random.uniform(*bounds['C'])
        gamma = np.random.uniform(*bounds['gamma'])

        params = np.array([C, gamma])

        fit = objective(
            params,
            X_train_scaled,
            y_train_scaled,
            X_test_scaled,
            y_test_scaled
        )

        solutions.append(params)
        fitnesses.append(fit)

    solutions = np.array(solutions)
    fitnesses = np.array(fitnesses)

    # ==========================================
    # PARETO ARCHIVE
    # ==========================================
    non_dominated = []

    for i, f in enumerate(fitnesses):

        dominated = False

        for j, other in enumerate(fitnesses):

            if i != j and dominates(other, f):
                dominated = True
                break

        if not dominated:
            non_dominated.append(i)

    archive = solutions[non_dominated]
    archive_f = fitnesses[non_dominated]

    # ==========================================
    # AMBIL SOLUSI DENGAN MSE TERKECIL
    # ==========================================
    best_idx = np.argmin(archive_f[:, 0])

    best_params = archive[best_idx]
    best_mse = archive_f[best_idx, 0]
    best_nsv = archive_f[best_idx, 1]

    return best_params, best_mse, best_nsv


# ==========================================================
# 10 INDEPENDENT RUNS
# ==========================================================
n_runs = 10

mse_list = []
nsv_list = []
best_params_list = []

print("===== RANDOM SEARCH SVR (MSE & #SV) =====")

for run in range(n_runs):

    best_params, best_mse, best_nsv = random_search_svr(
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        y_test_scaled,
        iter_max=100
    )

    mse_list.append(best_mse)
    nsv_list.append(best_nsv)
    best_params_list.append(best_params)

    print(
        f"Run {run+1:02d} | "
        f"Best C={best_params[0]:.4f}, "
        f"gamma={best_params[1]:.4f}, "
        f"MSE={best_mse:.5f}, "
        f"#SV={int(best_nsv)}"
    )

# ==========================================================
# REKAP
# ==========================================================
print("\n===== SUMMARY =====")
print(f"Mean MSE  : {np.mean(mse_list):.5f}")
print(f"Std MSE   : {np.std(mse_list):.5f}")
print(f"Mean #SV  : {np.mean(nsv_list):.2f}")
print(f"Std #SV   : {np.std(nsv_list):.2f}")


# In[ ]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

# ==========================================================
# OBJECTIVE FUNCTION
# ==========================================================
def objective(params,
              X_train_scaled, y_train_scaled,
              X_test_scaled, y_test_scaled):

    C, gamma = params

    model = SVR(
        C=C,
        gamma=gamma,
        kernel='rbf'
    )

    model.fit(X_train_scaled, y_train_scaled)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test_scaled, y_pred)
    n_sv = len(model.support_)

    return mse, n_sv


# ==========================================================
# DOMINATION CHECK
# ==========================================================
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


# ==========================================================
# TOURNAMENT SELECTION
# ==========================================================
def tournament_selection(population, fitness, k=3):

    idx = np.random.choice(len(population), k, replace=False)

    best = idx[0]

    for i in idx[1:]:
        if dominates(fitness[i], fitness[best]):
            best = i

    return population[best].copy()


# ==========================================================
# GENETIC ALGORITHM SVR
# ==========================================================
def ga_svr(X_train_scaled, y_train_scaled,
           X_test_scaled, y_test_scaled,
           pop_size=40,
           iter_max=100,
           crossover_rate=0.9,
           mutation_rate=0.1):

    bounds = {
        'C': (0.1, 10),
        'gamma': (0.1, 10)
    }

    # ======================================================
    # INITIAL POPULATION
    # ======================================================
    population = np.array([
        [
            np.random.uniform(*bounds['C']),
            np.random.uniform(*bounds['gamma'])
        ]
        for _ in range(pop_size)
    ])

    archive = []
    archive_f = []

    # ======================================================
    # GENERATIONS
    # ======================================================
    for gen in range(iter_max):

        fitness = np.array([
            objective(
                ind,
                X_train_scaled,
                y_train_scaled,
                X_test_scaled,
                y_test_scaled
            )
            for ind in population
        ])

        # ----------------------------------------------
        # Update archive
        # ----------------------------------------------
        if len(archive) == 0:
            all_pop = population
            all_fit = fitness
        else:
            all_pop = np.vstack((np.array(archive), population))
            all_fit = np.vstack((np.array(archive_f), fitness))

        non_dominated = []

        for i, f in enumerate(all_fit):

            dominated = False

            for j, other in enumerate(all_fit):

                if i != j and dominates(other, f):
                    dominated = True
                    break

            if not dominated:
                non_dominated.append(i)

        archive = all_pop[non_dominated]
        archive_f = all_fit[non_dominated]

        # ----------------------------------------------
        # Create offspring
        # ----------------------------------------------
        offspring = []

        while len(offspring) < pop_size:

            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)

            child1 = parent1.copy()
            child2 = parent2.copy()

            # Crossover
            if np.random.rand() < crossover_rate:

                alpha = np.random.rand()

                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = alpha * parent2 + (1 - alpha) * parent1

            # Mutation
            for child in [child1, child2]:

                if np.random.rand() < mutation_rate:
                    child[0] += np.random.normal(0, 0.5)

                if np.random.rand() < mutation_rate:
                    child[1] += np.random.normal(0, 0.5)

                child[0] = np.clip(child[0], *bounds['C'])
                child[1] = np.clip(child[1], *bounds['gamma'])

                offspring.append(child)

                if len(offspring) >= pop_size:
                    break

        population = np.array(offspring)

    # ======================================================
    # FINAL RESULT
    # ======================================================
    best_idx = np.argmin(np.array(archive_f)[:, 0])

    best_params = np.array(archive)[best_idx]
    best_mse = np.array(archive_f)[best_idx, 0]
    best_nsv = np.array(archive_f)[best_idx, 1]

    return best_params, best_mse, best_nsv


# ==========================================================
# 10 INDEPENDENT RUNS
# ==========================================================
n_runs = 10

mse_list = []
nsv_list = []
best_params_list = []

print("===== GA-SVR (MSE & #SV) =====")

for run in range(n_runs):

    best_params, best_mse, best_nsv = ga_svr(
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        y_test_scaled,
        pop_size=40,
        iter_max=100
    )

    mse_list.append(best_mse)
    nsv_list.append(best_nsv)
    best_params_list.append(best_params)

    print(
        f"Run {run+1:02d} | "
        f"Best C={best_params[0]:.4f}, "
        f"gamma={best_params[1]:.4f}, "
        f"MSE={best_mse:.5f}, "
        f"#SV={int(best_nsv)}"
    )

print("\n===== SUMMARY =====")
print(f"Mean MSE : {np.mean(mse_list):.5f}")
print(f"Std MSE  : {np.std(mse_list):.5f}")
print(f"Mean #SV : {np.mean(nsv_list):.2f}")
print(f"Std #SV  : {np.std(nsv_list):.2f}")


# In[26]:



def svr_baseline(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
                 C=1, epsilon=2.842257061733180, gamma=0.12219904502204129, kernel='rbf'):

    model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')
    model.fit(X_train_scaled, y_train_scaled)

    # prediksi
    y_pred = model.predict(X_test_scaled)

    # hitung metrik
    mse = mse_score(y_test_scaled, y_pred)
    d = dstat(y_test_scaled, y_pred)

    # jumlah support vector
    num_sv = model.n_support_.sum()

    print("=== HASIL SVR BIASA ===")
    print(f"C = {C}, gamma = {gamma}, epsilon = {epsilon}")
    print(f"MSE   = {mse:.6f}")
    print(f"Dstat = {d:.4f}")
    print(f"Support Vector = {num_sv}")

    # return 5 nilai
    return mse, d, num_sv, model, y_pred


# In[31]:


mse, d, num_sv, model, y_pred = svr_baseline(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)


# In[40]:


import matplotlib.pyplot as plt
import numpy as np

# Nomor data di test set
x_index = np.arange(len(y_test))

plt.figure(figsize=(10,5))

# Plot actual
plt.plot(x_index, y_test, 
         marker='s', markersize=5, linestyle='-', 
         color='blue', alpha=0.7, label='Actual (y_test)')

# Plot predicted
plt.plot(x_index, y_pred, 
         marker='o', markersize=5, linestyle='--', 
         color='red', alpha=0.7, label='Predicted (SVR)')

plt.xlabel("Index Data (Test Set)")
plt.ylabel("Value")
plt.title("Prediction on Test Set (SVR)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[50]:


import numpy as np

def sensitivity_analysis_permutation(model, X_test_scaled, y_test, metric=mse_score):
    base_pred = model.predict(X_test_scaled)
    base_mse = metric(y_test, base_pred)

    importances = []

    for i in range(X_test_scaled.shape[1]):
        X_permuted = X_test_scaled.copy()
        np.random.shuffle(X_permuted[:, i])  # acak fitur ke-i

        y_pred_perm = model.predict(X_permuted)
        perm_mse = metric(y_test, y_pred_perm)

        importance = perm_mse - base_mse
        importances.append(importance)

        print(f"Fitur {i} → ΔMSE = {importance:.6f}")

    return np.array(importances)


# In[51]:


def sensitivity_analysis_permutation(model, X_test_scaled, y_test, metric=mse_score):

    # pastikan numpy array
    X_test_np = np.asarray(X_test_scaled)

    base_pred = model.predict(X_test_np)
    base_mse = metric(y_test, base_pred)

    importances = []

    for i in range(X_test_np.shape[1]):
        X_permuted = X_test_np.copy()
        np.random.shuffle(X_permuted[:, i])  # aman karena sudah numpy

        y_pred_perm = model.predict(X_permuted)
        perm_mse = metric(y_test, y_pred_perm)

        importance = perm_mse - base_mse
        importances.append(importance)

        print(f"Fitur {i} → ΔMSE = {importance:.6f}")

    return np.array(importances)


# In[52]:


mse, d, num_sv, model, y_pred = svr_baseline(X_train_scaled, y_train, X_test_scaled, y_test)
importances = sensitivity_analysis_permutation(model, X_test_scaled, y_test)


# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sensitivity_analysis_permutation_full(model, X_test_scaled, y_test, feature_names=None, metric=mse_score):
    # pastikan numpy array
    X_test_np = np.asarray(X_test_scaled)

    # nama fitur
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_test_np.shape[1])]

    # baseline MSE
    base_pred = model.predict(X_test_np)
    base_mse = metric(y_test, base_pred)

    importances = []

    # shuffle per fitur
    for i in range(X_test_np.shape[1]):
        X_permuted = X_test_np.copy()
        np.random.shuffle(X_permuted[:, i])

        y_pred_perm = model.predict(X_permuted)
        perm_mse = metric(y_test, y_pred_perm)

        delta = perm_mse - base_mse
        importances.append(delta)

        print(f"{feature_names[i]} → ΔMSE = {delta:.6f}")

    # buat tabel
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Delta_MSE": importances
    }).sort_values("Delta_MSE", ascending=False)

    print("\n=== RANKING FITUR (lebih besar = lebih penting) ===")
    print(df_importance)

    # plot
    plt.figure(figsize=(8,4))
    plt.bar(df_importance["Feature"], df_importance["Delta_MSE"])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("ΔMSE")
    plt.title("Sensitivity Analysis – Permutation Importance (SVR)")
    plt.tight_layout()
    plt.show()

    return df_importance


# In[54]:


mse, d, num_sv, model, y_pred = svr_baseline(X_train_scaled, y_train, X_test_scaled, y_test)

# Kalau dataset punya nama kolom:
feature_names = X_train_scaled.columns

df_importance = sensitivity_analysis_permutation_full(
    model,
    X_test_scaled,
    y_test,
    feature_names=feature_names
)


# In[ ]:




