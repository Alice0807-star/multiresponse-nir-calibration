"""Candidate-conditioned task attention, supervised by actual three-target gains.

No global attention averaging, inverse-weight deletion, Q bootstrap or child
reward in the input. The identical forward path ranks proposals in learning
and deployment. Current prototype is an offline mechanism test only.
"""
import math
import torch
from torch import nn

QUERY_NAMES=['paid_parent_1_minus_R2','fit_residual_MSE','add_fit_residual_cov','remove_fit_residual_cov',
    'add_fit_target_corr','remove_fit_target_corr','remove_stdX_coefficient','mean_abs_parent_coefficient',
    'add_remove_fit_corr','max_add_survivor_abs_corr','mean_add_survivor_abs_corr',
    'add_column_coordinate','remove_column_coordinate','mean_survivor_target_corr']
TOKEN_NAMES=['add_survivor_fit_corr','remove_survivor_fit_corr','survivor_target_corr','survivor_residual_cov',
    'survivor_stdX_coefficient','abs_survivor_stdX_coefficient','survivor_column_coordinate','survivor_mean_other_abs_corr']

class ConditionalTaskAttention(nn.Module):
    def __init__(self,dimension=32):
        super().__init__();self.dimension=dimension
        self.query=nn.Linear(len(QUERY_NAMES),dimension);self.key=nn.Linear(len(TOKEN_NAMES),dimension);self.value=nn.Linear(len(TOKEN_NAMES),dimension)
        self.task_embedding=nn.Parameter(torch.randn(3,dimension)*.05)
        self.task_relation=nn.MultiheadAttention(dimension,4,batch_first=True,dropout=0.)
        self.norm=nn.LayerNorm(dimension)
    def forward(self,query,tokens):
        if query.shape[1:]!=(3,len(QUERY_NAMES)) or tokens.shape[1:]!=(3,15,len(TOKEN_NAMES)):raise ValueError('Conditional features contract')
        q=torch.tanh(self.query(query)+self.task_embedding)
        k=torch.tanh(self.key(tokens));v=torch.tanh(self.value(tokens))
        weights=(q[:,:,None,:]*k).sum(-1)/math.sqrt(self.dimension);weights=weights.softmax(-1)
        context=(weights[:,:,:,None]*v).sum(2)
        state=q+context;exchange,relation=self.task_relation(state,state,state,need_weights=True)
        return context,self.norm(state+exchange),weights,relation

class ConditionalGainModel(nn.Module):
    kind='conditional_three_target_gain_attention_v1'
    def __init__(self,*,adaptive=True,dimension=32):
        super().__init__();self.attention=ConditionalTaskAttention(dimension)
        self.head=nn.Sequential(nn.Linear(len(QUERY_NAMES)+2*dimension,64),nn.GELU(),nn.Linear(64,1))
        for p in self.attention.parameters():p.requires_grad_(adaptive)
        self.adaptive=adaptive;self.dimension=dimension
    def forward(self,query,tokens,*,diagnostics=False):
        context,shared,weights,relation=self.attention(query,tokens)
        prediction=self.head(torch.cat([query,context,shared],-1)).squeeze(-1)*.01
        if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite conditional gain')
        return (prediction,weights,relation) if diagnostics else prediction
    def rank(self,query,tokens):
        return self(query,tokens).mean(-1)

def actual_gain_feedback(prediction,true_gain):
    # One common R2 scale for all targets; main reward remains uniform mean R2.
    return ((prediction-true_gain)/.01).square().mean()
