"""
Be lazy because there are 2 days til submission
"""
import pandas as pd
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from sklearn.metrics import log_loss
import tensorflow as tf
import mlflow
import argparse
from typing import List

ap = argparse.ArgumentParser()
ap.add_argument("--holdout", dest="holdout", default="2018,2019")
ap.add_argument("--lr", dest="lr", default=0.001, type=float)
ap.add_argument("--cap", dest="cap", default=0.04, type=float)
ap.add_argument("--epochs", dest="epochs", default=5000, type=int)
ap.add_argument("--point-noise", dest="point_noise", default=1, type=int)
ap.add_argument("--num-copies", dest="n_copies", default=2, type=int)
# ap.add_argument("--loc-weight", dest="loc_weight", default=0.1, type=float)
# ap.add_argument("--scale-weight", dest="scale_weight", default=0.1, type=float)
args = ap.parse_args()

mlflow.set_tracking_uri("sqlite:///tracking.db")
mlflow.tensorflow.autolog()

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

year_goals = {2018: (0.53, 0.55), 2017: (0.438, 0.467), 2019: (0.414, 0.437)}

TO_EXCLUDE = [
    "T1Score",
    "DayNum",
    "T2Score",
    "T1TeamID",
    "T2TeamID",
    "T1Loc",
    "T2Loc",
    "NumOT",
    "Season",
    "diff",
]


def pom_rename(df: pd.DataFrame, wl: str, exclude: List[str] = ["Season", "TeamID"]):
    """
    Rename the pom columns based on whether you're joining to the winning or losing team ID
    """
    return df.rename(columns={c: wl + c for c in df.columns})


def df_rename(df: pd.DataFrame, t1_prefix: str, t2_prefix: str):
    """
    Rename columns based on whether we care about the winning or losing team
    """
    renames = {}
    for col in df.columns:
        if col[0] == t1_prefix:
            renames[col] = "T1" + col[1:]
        if col[0] == t2_prefix:
            renames[col] = "T2" + col[1:]
    return df.rename(columns=renames)


curr_year = 2021
start_year = 2011
# cap = 0.04
holdout = [int(i) for i in args.holdout.split(",")] if args.holdout is not None else []
# epochs = 2000
base = "mens/"
model_type = "variational"
goal_low, goal_high = [], []
goal = ""
for year in holdout:
    g = year_goals.get(year)
    if g is not None:
        goal_low.append(g[0])
        goal_high.append(g[1])
if len(goal_high) > 0:
    goal_list = [np.mean(goal_low), np.mean(goal_high)]
    goal = str(goal_list[0]) + " - " + str(goal_list[1])

mlflow.log_params(vars(args))

pom_path = base + "pom_output.csv"
tourney_path = base + "MDataFiles_Stage2/MNCAATourneyCompactResults.csv"
sub_path = base + "MDataFiles_Stage2/MSampleSubmissionStage2.csv"

pom = pd.read_csv(pom_path).set_index(["Season", "TeamID"])
mlflow.log_metric("n_pom_records", pom.shape[0])
pom = pom.drop_duplicates()
mlflow.log_metric("n_pom_records", pom.shape[0])

tourneys = pd.read_csv(tourney_path)
sample_sub = pd.read_csv(sub_path)[["ID"]]

tourneys = tourneys[tourneys.Season >= start_year]
sample_sub["Season"] = curr_year
sample_sub["Team1ID"] = sample_sub.ID.apply(lambda x: int(x[5:9]))
sample_sub["Team2ID"] = sample_sub.ID.apply(lambda x: int(x[10:14]))


joined = pd.merge(
    tourneys,
    pom_rename(pom, "W"),
    left_on=["Season", "WTeamID"],
    right_index=True,
    how="inner",
)
joined = pd.merge(
    joined,
    pom_rename(pom, "L"),
    left_on=["Season", "LTeamID"],
    right_index=True,
    how="inner",
)
joined["diff"] = joined["WScore"] - joined["LScore"]
joined["diff"] = joined[["diff", "NumOT"]].apply(
    lambda x: np.min([x["diff"], 3]) if x["NumOT"] > 0 else x, axis=1
)

mlflow.log_metric("n_samples_sub", sample_sub.shape[0])
sample_sub = pd.merge(
    sample_sub,
    pom_rename(pom, "T1"),
    left_on=["Season", "Team1ID"],
    right_index=True,
    how="inner",
)
mlflow.log_metric("n_samples_sub", sample_sub.shape[0])
sample_sub = pd.merge(
    sample_sub,
    pom_rename(pom, "T2"),
    left_on=["Season", "Team2ID"],
    right_index=True,
    how="inner",
)
mlflow.log_metric("n_samples_sub", sample_sub.shape[0])


pos = df_rename(joined, "W", "L")
neg = df_rename(joined, "L", "W")
neg["diff"] = -neg["diff"]

combined = pos.append(neg)
# copied = combined.copy()
# for n in range(args.n_copies - 1):
#     new_copy = copied.copy()
#     new_copy["diff"] = new_copy["diff"] + np.random.randint(
#         -args.point_noise, args.point_noise + 1, size=(new_copy.shape[0])
#     )
#     combined = combined.append(new_copy)

train = combined[~combined.Season.isin(holdout)].copy()
dev = combined[combined.Season.isin(holdout)].copy()

features = [c for c in combined.columns if c not in TO_EXCLUDE]
x_train = train[features].values
x_dev = dev[features].values
y_train = train["diff"].values.astype(np.float32)
y_dev = dev["diff"].values.astype(np.float32)
y_train_wins = (y_train > 0) * 1.0
y_dev_wins = (y_dev > 0) * 1.0

x_test = sample_sub[features].values.astype(np.float32)


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(
                        loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])
                    ),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1
                )
            ),
        ]
    )


def to_pred(y_dist, samp=500, cap=0.04):
    s = y_dist.sample(samp).numpy()
    s2 = np.where(s > 0, 1, 0)
    s3 = np.mean(s2, axis=0)
    return np.clip(s3, cap, 1 - cap)


def get_mean(y_dist, samp=500):
    s = y_dist.sample(samp).numpy()
    return np.mean(s, axis=0)


class EvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data=(), goal=None, interval=10):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.goal = goal

    def on_epoch_end(self, epoch, logs={}):
        vs_goal = str(self.goal)
        if self.X_val.shape[0] > 0:
            y_pred = self.model(self.X_val)
            y_pred = to_pred(y_pred, cap=args.cap)
            score = log_loss(self.y_val, y_pred)
            mlflow.log_metric("val_log_loss", score, step=epoch)
            print("epoch: {:d} - val_log_loss: {:.6f} {}".format(epoch, score, vs_goal))


model = tf.keras.Sequential(
    [
        tfp.layers.DenseVariational(
            1 + 1, posterior_mean_field, prior_trainable, kl_weight=1 / x_train.shape[0]
        ),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(
                loc=t[..., :1],
                scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]),
            )
        ),
    ]
)

model.compile(optimizer=tf.optimizers.Adam(learning_rate=args.lr), loss=negloglik)
model.fit(
    x_train,
    y_train,
    epochs=args.epochs,
    callbacks=[EvaluationCallback((x_dev, y_dev_wins), goal=goal)],
)

sub_pred = model(x_test)
pred_test = to_pred(sub_pred, cap=args.cap)

sample_sub["Pred"] = pred_test
output = sample_sub[["ID", "Pred"]]

output.to_csv("preds.csv", index=False)
mlflow.log_artifact("preds.csv")

mean_test = get_mean(sub_pred)
sample_sub["Pred"] = mean_test
output = sample_sub[["ID", "Pred"]]

output.to_csv("spreads.csv", index=False)
mlflow.log_artifact("spreads.csv")