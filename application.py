import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pingouin

import jax.numpy as jnp
from jax import random, vmap

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
sns.set()

sheet_url = "https://docs.google.com/spreadsheets/d/1pdWo_Q4FyHkL_E12zrp63HQA3txPs4TNGP2MgvO3Kfk/export?format=csv&id=1pdWo_Q4FyHkL_E12zrp63HQA3txPs4TNGP2MgvO3Kfk&gid=0"
    
data = pd.read_csv(sheet_url,index_col=0)
data.columns = data.columns.str.strip(' ')


repl = lambda m: m.group(0)

for col in data.columns:
    if data[col].dtype == object:
        if data[col].str.contains('[0-9]').any():
            data[col] = data[col].str.replace('[^0-9]',"").astype(int)

# 2015, 2016, 2017, 2018, 2019
# Normalised to 1 EA member in 2019

# Calculates the remaining members who joined in each year since 2015 given 
#  - The hazard rates for each cohort (annual % chance of dropping out after being involved i years)
#  - The total number who joined in each year since 2015
#  - A final year of y
def member_decay(hazards, joineds, y):
    cum_hazards = jnp.array([(1-hazards[:i]).prod() for i in range(y)])[::-1]
    return cum_hazards*joineds[:y]

# Calculates the total members who joined in each year since 2015 given 
#  - The hazard rates for each cohort (annual % chance of dropping out after being involved i years)
#  - The remaining number who joined in each year since 2015
#  - A final year of y
def inverse_member_decay(hazards, remaining, y):
    cum_hazards = jnp.array([(1-hazards[:i]).prod() for i in range(y)])[::-1]
    return remaining[:y]/cum_hazards

EA_TOTAL = jnp.array([2395, 2916, 4150, 4587, 5669])/5669
#EA_TOTAL = jnp.array([1044,2395, 2916, 4150, 4587, 5669])/5669

# Split the 2015 cohort
REMAINING_2019 = data['When joined EA'].value_counts()[::-1].values/2000
# REMAINING_2019 = np.array([REMAINING_2019[0]*1/3] + [REMAINING_2019[0]*2/3] + list(REMAINING_2019[1:]))

TEST_HAZARDS = jnp.array([1/3,1/5,1/6,1/7]) # For testing purposes only
TEST_JOINED = inverse_member_decay(TEST_HAZARDS,REMAINING_2019,5) # For testing purposes only

# Model of member counts and dropout
# The relationship between total number joined each year and remaining cohort from that year is given by functions
# above

# Also assume that there's noise in actual number who join, reported member totals and reported cohort size
# Just set noise to normal distribution with 1SD = 10% reported value in each case, didn't think about it too hard
# The model doesn't converge if I try to use sampling error for the noise, which isn't appropriate anyway
# Because GiveWell donors and survey respondents aren't from the same population

def model(obs_members = None, obs_remaining = None,beta_c1 = None, beta_c2 = None):
    hazards = numpyro.sample('hazard',dist.Beta(beta_c1,beta_c2)) #needs an additional component if you want to split 2015
    joined_est = inverse_member_decay(hazards,obs_remaining,len(obs_remaining))
    joined = numpyro.sample('joined', dist.Normal(joined_est,joined_est*0.1))
    members = jnp.array([member_decay(hazards, joined, y).sum() for y in range(1,1+len(obs_members))])
    meas_members = numpyro.sample('meas_members',dist.Normal(members,members*0.1),obs=obs_members)
    remaining = member_decay(hazards, joined, len(obs_members))
    reported_remain = numpyro.sample('reported_remaining',dist.Normal(remaining,remaining*0.1),obs=obs_remaining)

def prior_only(obs_members = None, obs_remaining = None, beta_c1 = None, beta_c2 = None):
    hazards = numpyro.sample('hazard',dist.Beta(beta_c1,beta_c2))
    joined_est = inverse_member_decay(hazards,obs_remaining,len(obs_remaining))
    joined = numpyro.sample('joined', dist.Normal(joined_est,joined_est*0.1))
    
def run_samples(model,
                obs_members = EA_TOTAL, 
                obs_remaining = REMAINING_2019,
                beta_c1 = [2,2,2,2],
                beta_c2 = [4,8,10,12],
                num_warmup = 2000, 
                num_samples = 2000):
    # Start from this source of randomness. We will split keys for subsequent operations.
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    # Run NUTS.
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, obs_members = obs_members, obs_remaining = obs_remaining, beta_c1 = beta_c1, beta_c2 = beta_c2)
    mcmc.print_summary()
    samples_1 = mcmc.get_samples()
    return samples_1

samples = run_samples(model)
samples_prior = run_samples(prior_only)
samples_flat = run_samples(model, beta_c2 = [4,4,4,4])
samples['cumulative'] =  jnp.cumprod(1-samples['hazard'],axis=1)

hazard_rates = pd.melt(pd.DataFrame(samples['hazard']),
                       value_vars=[0,1,2,3],
                       var_name = 'years involved', 
                       value_name = 'annual probability of dropout')

hazard_prior = pd.melt(pd.DataFrame(samples_prior['hazard']),
                       value_vars=[0,1,2,3],
                       var_name = 'years involved', 
                       value_name = 'annual probability of dropout')
hazard_flat = pd.melt(pd.DataFrame(samples_flat['hazard']),
                       value_vars=[0,1,2,3],
                       var_name = 'years involved', 
                       value_name = 'annual probability of dropout')

cumulative = pd.melt(pd.DataFrame(samples['cumulative']),
                       value_vars=[0,1,2,3],
                       var_name = 'years involved', 
                       value_name = 'probability of remaining')
cumulative['years involved']+= 1

hazard_rates['source'] = 'posterior'
hazard_prior['source'] = 'prior'
hazard_flat['source'] = 'posterior from flat prior'


hazard_rates = hazard_rates.append(hazard_prior).append(hazard_flat)

summary = hazard_rates[(hazard_rates['source']=='posterior')]
noflat = hazard_rates[hazard_rates['source']!='posterior from flat prior']
noprior = hazard_rates[hazard_rates['source']!='prior']
fullcohort = pd.melt(pd.DataFrame(samples['joined']).rename(columns = {0:2015,1:2016,2:2017,3:2018,4:2019})*2000,
                       value_vars=[2015,2016,2017,2018,2019],
                       var_name = 'years involved', 
                       value_name = 'relative number joined')