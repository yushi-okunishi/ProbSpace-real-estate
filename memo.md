# 実験録

## simple light gbm

benchmarkにする

2020/04/16

published_land_priceを使わず
object featureをlableencodingして、そのままlightgbmに入れた

lightgbmでカラム名が英語は使えない? カラム名を英語に書き換える

lgb_param = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.4,
    "num_iteration": 100000,
    "early_stopping_round": 500,
}

5 foldでのcvスコア
final score: 239.09118787952966

cvのスコアより、lbのスコアのほうがかなり良い、90.0くらい

<考察>
Public LBには高額の外れ値がすくないから、cvでの結果より良いスコアになっている？(discussionより)
RMSEは外れ値に弱いので、如何に外れ値を予測できるかがキモ？
過去のBoston住宅価格予測を復習すれば、何か良い情報があるかも？


## Published_land_priceのmergeをためす

published_land_priceには、ある不動産の政府によって公開されている価格がある。
フルの住所に対して、価格が掲載されているのに対して、train.csvには、地区名までしか存在しない

住所のフォーマット <県名> <市区町村> <地区> <番地>

trainに入っているのに、publishedには以下の市区町村が入っていない
13307    西多摩郡檜原村
13308    西多摩郡奥多摩町

publishedの住所から、地区を抽出して、ある地区の、用途、種類、面積ごとの値段をtarget encodingしたい

false -> true
田市木曽西 -> 木曽西

## 04/22 再開

https://prob.space/topics/Ni-sla-Post98c03996cff12e04067f
用途のBoWエンコーディング
のちに入れる

## 04/23
area, floor_area, frontage, transaction_pointに前処理
final score: 226.77181050363598
少し更新

floor plan 前処理
final score: 226.2752422985371

nearest station distance
year of construction 
前処理
final score: 227.92646326131916

learning rate 0.4 -> 0.04
final score: 218.5316175320758

targetをnp.log1p()して予測、結果をnp.expm1()
これは訓練が重いので、以降のスコアで採用しない
final score: 203.2333139640694

preprocess use for
final score: 221.7483900781018

lgb tunerで最適化
final score: 213.61388036943623

building structure, transaction point, 前処理
final score: 214.89878988981826

categorical featureについて、target encoding, target std encoding
final score: 210.714

上のをlog1p()した
final score: 201.28845295497126

frontageとfront_road_widthにnullがある
回帰で埋めてからlinear regression?またはlog1_areaだけでtarget_regression?

linear regressionでyもscalingしてから予測する

# 締め切り

2020/04/27 0:00 JST