from sklearn.preprocessing import MinMaxScaler

class MinMaxScaler(MinMaxScaler):
    """Standardize features by removing the mean and scaling to unit variance.
    The standard score of a sample `x` is calculated as:
        z = (x - u) / s
    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.
    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using
    :meth:`transform`.
    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual features do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).
    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    than others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.
    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.
    Read more in the :ref:`User Guide <preprocessing_scaler>`.
    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.
    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.
    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).
    Attributes
    ----------
    scale_ : ndarray of shape (n_features,) or None
        Per feature relative scaling of the data to achieve zero mean and unit
        variance. Generally this is calculated using `np.sqrt(var_)`. If a
        variance is zero, we can't achieve unit variance, and the data is left
        as-is, giving a scaling factor of 1. `scale_` is equal to `None`
        when `with_std=False`.
        .. versionadded:: 0.17
           *scale_*
    mean_ : ndarray of shape (n_features,) or None
        The mean value for each feature in the training set.
        Equal to ``None`` when ``with_mean=False``.
    var_ : ndarray of shape (n_features,) or None
        The variance for each feature in the training set. Used to compute
        `scale_`. Equal to ``None`` when ``with_std=False``.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    n_samples_seen_ : int or ndarray of shape (n_features,)
        The number of samples processed by the estimator for each feature.
        If there are no missing samples, the ``n_samples_seen`` will be an
        integer, otherwise it will be an array of dtype int. If
        `sample_weights` are used it will be a float (if no missing data)
        or an array of dtype float that sums the weights seen so far.
        Will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.
    See Also
    --------
    scale : Equivalent function without the estimator API.
    :class:`~sklearn.decomposition.PCA` : Further removes the linear
        correlation across features with 'whiten=True'.
    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.
    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> scaler = StandardScaler()
    >>> print(scaler.fit(data))
    StandardScaler()
    >>> print(scaler.mean_)
    [0.5 0.5]
    >>> print(scaler.transform(data))
    [[-1. -1.]
     [-1. -1.]
     [ 1.  1.]
     [ 1.  1.]]
    >>> print(scaler.transform([[2, 2]]))
    [[3. 3.]]
    """
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        super(MinMaxScaler, self).__init__(feature_range=feature_range, copy=copy, clip=clip)
        
        
    def column_index(self, query_cols):
        cols = self.feature_names_in_
        sidx = np.argsort(cols)
        return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
        
    def transform(self,X):
        """Scale features of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        if type(X) == pd.DataFrame:
            icol = self.column_index(X.columns)
            X = X*self.scale_[icol] + self.min_[icol]
            if self.clip:
                np.clip(X, self.feature_range[0], self.feature_range[1], out=X)
            return X
        else:
            return super(MinMaxScaler, self).transform(X)

        
    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        if type(X) == pd.DataFrame:
            icol = self.column_index(X.columns)
            X = (X - self.min_[icol])/self.scale_[icol]
            return X
        else:
            return super(MinMaxScaler, self).inverse_transform(X)




from sklearn.preprocessing import StandardScaler

class StandardScaler(StandardScaler):
    """Transform features by scaling each feature to a given range.
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.
    The transformation is given by::
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
    where min, max = feature_range.
    This transformation is often used as an alternative to zero mean,
    unit variance scaling.
    Read more in the :ref:`User Guide <preprocessing_scaler>`.
    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).
    clip : bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `feature range`.
        .. versionadded:: 0.24
    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per feature adjustment for minimum. Equivalent to
        ``min - X.min(axis=0) * self.scale_``
    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data. Equivalent to
        ``(max - min) / (X.max(axis=0) - X.min(axis=0))``
        .. versionadded:: 0.17
           *scale_* attribute.
    data_min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data
        .. versionadded:: 0.17
           *data_min_*
    data_max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the data
        .. versionadded:: 0.17
           *data_max_*
    data_range_ : ndarray of shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data
        .. versionadded:: 0.17
           *data_range_*
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    n_samples_seen_ : int
        The number of samples processed by the estimator.
        It will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    See Also
    --------
    minmax_scale : Equivalent function without the estimator API.
    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    Examples
    --------
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    >>> scaler = MinMaxScaler()
    >>> print(scaler.fit(data))
    MinMaxScaler()
    >>> print(scaler.data_max_)
    [ 1. 18.]
    >>> print(scaler.transform(data))
    [[0.   0.  ]
     [0.25 0.25]
     [0.5  0.5 ]
     [1.   1.  ]]
    >>> print(scaler.transform([[2, 2]]))
    [[1.5 0. ]]
    """
    def __init__(self, copy=True, with_mean=True, with_std=True):
        super(StandardScaler, self).__init__(copy=copy, with_mean=with_mean, with_std=with_std)
        
    def column_index(self, query_cols):
        cols = self.feature_names_in_
        sidx = np.argsort(cols)
        return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
        
    def transform(self, X, copy=None):
        """Perform standardization by centering and scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        if type(X) == pd.DataFrame:
            icol = self.column_index(X.columns)
            if self.with_mean:
                X = X - self.mean_[icol]
            if self.with_std:
                X = X/self.scale_[icol]
            return X
        else:
            return super(StandardScaler, self).transform(X,copy=copy)
        
    def inverse_transform(self, X, copy=None):
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        if type(X) == pd.DataFrame:
            icol = self.column_index(X.columns)
            if self.with_std:
                X = X*self.scale_[icol]
            if self.with_mean:
                X = X+self.mean_
            return X
        else:
            return super(StandardScaler, self).inverse_transform(X,copy=copy)
