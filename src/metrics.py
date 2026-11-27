"""Shared descriptive agreement metrics used by every paper evaluation."""
import numpy as np

def _array(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.asarray(value,dtype=np.float64).reshape(-1)

def icc_a1(first,second):
    """Single-measure absolute-agreement ICC(A,1), complete paired observations.

    Error mean square is SSE / ((n-1)*(k-1)); it is not the sample variance
    of the residual matrix. Nonfinite input is rejected rather than omitted.
    """
    first,second=_array(first),_array(second)
    if first.shape!=second.shape:raise ValueError('Paired ICC lengths differ')
    X=np.column_stack([first,second]);n,k=X.shape
    if n<2 or not np.isfinite(X).all():return float('nan')
    rows=X.mean(axis=1);columns=X.mean(axis=0);grand=X.mean()
    MSR=k*np.sum((rows-grand)**2)/(n-1)
    MSC=n*np.sum((columns-grand)**2)/(k-1)
    residual=X-rows[:,None]-columns[None,:]+grand
    MSE=np.sum(residual**2)/((n-1)*(k-1))
    denominator=MSR+(k-1)*MSE+k*(MSC-MSE)/n
    return float((MSR-MSE)/denominator) if denominator!=0 else float('nan')
