"""
This file is from Lenskit. This file needs to be modified before generating recommendation lenskit.
It contains the added code that is needed to upload the new candidates
"""

import logging
import warnings
import json
import pandas as pd
import numpy as np
import random
from ..algorithms import Recommender
from .. import util
from ..sharing import PersistedModel

_logger = logging.getLogger(__name__)


def _recommend_user(algo, req):
    user, n, candidates = req

    _logger.debug('generating recommendations for %s', user)
    watch = util.Stopwatch()
    res = algo.recommend(user, n, candidates)
    _logger.debug('%s recommended %d/%s items for %s in %s',
                  str(algo), len(res), n, user, watch)

    res['user'] = user
    res['rank'] = np.arange(1, len(res) + 1)

    return res.reset_index(drop=True)


def __standard_cand_fun(candidates):
    """
    Convert candidates from the forms accepted by :py:fun:`recommend` into
    a standard form, a function that takes a user and returns a candidate
    list.
    This function is modified by us to be able to read our own candidates
    """
    candidates = dict()
    # print ("test")
    with open('/content/gdrive/MyDrive/Colab Notebooks/PerBlur_Lkpy_Experiments/Flixster/FX_ExceptTestSet_Candidate_Items_UnratedIntersection_All_Except_Imputation_top50.json') as json_file:
        data = json.load(json_file)

    for user, v in data.items():
        candidates[user] = list(random.sample(v, 1000))  # flixster 100, ml100k 850, ml1m 1000 np.random.shuffle(v) #

    # if isinstance(candidates, dict):
    #     return candidates.get
    # elif candidates is None:
    #     return lambda u: None
    # else:
    return candidates.get


def recommend(algo, users, n, candidates=0, *, n_jobs=None, **kwargs):
    """
    Batch-recommend for multiple users.  The provided algorithm should be a
    :py:class:`algorithms.Recommender`.

    Args:
        algo: the algorithm
        users(array-like): the users to recommend for
        n(int): the number of recommendations to generate (None for unlimited)
        candidates:
            the users' candidate sets. This can be a function, in which case it will
            be passed each user ID; it can also be a dictionary, in which case user
            IDs will be looked up in it.  Pass ``None`` to use the recommender's
            built-in candidate selector (usually recommended).
        n_jobs(int):
            The number of processes to use for parallel recommendations.  Passed to
            :func:`lenskit.util.parallel.invoker`.

    Returns:
        A frame with at least the columns ``user``, ``rank``, and ``item``; possibly also
        ``score``, and any other columns returned by the recommender.
    """

    if n_jobs is None and 'nprocs' in kwargs:
        n_jobs = kwargs['nprocs']
        warnings.warn('nprocs is deprecated, use n_jobs', DeprecationWarning)

    if not isinstance(algo, PersistedModel):
        rec_algo = Recommender.adapt(algo)
        if candidates is None and rec_algo is not algo:
            warnings.warn('no candidates provided and algo is not a recommender, unlikely to work')
        algo = rec_algo
        del rec_algo

    if 'ratings' in kwargs:
        warnings.warn('Providing ratings to recommend is not supported', DeprecationWarning)

    temp = {}
    candidates = __standard_cand_fun(temp)

    with util.parallel.invoker(algo, _recommend_user, n_jobs=n_jobs) as worker:
        _logger.info('recommending with %s for %d users (n_jobs=%s)',
                     str(algo), len(users), n_jobs)
        del algo
        timer = util.Stopwatch()
        results = worker.map((user, n, candidates(user)) for user in users)
        results = pd.concat(results, ignore_index=True, copy=False)
        _logger.info('recommended for %d users in %s', len(users), timer)

    return results
