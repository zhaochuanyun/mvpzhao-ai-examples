个人使用tensorflow、keras等框架进行训练的一些实例

# 1、验证码识别

`在2.7 GHz Intel Core i7的CPU上，能在分钟级别的时间内完成训练，成功率达到90%以上`

#### Step 1: Extract single letters from CAPTCHA images

Run: python3 datasets_generate.py

#### Step 2: Train the neural network to recognize single letters

Run: python3 train_cnn_model.py

#### Step 3: Use the model to solve CAPTCHAs!

Run: python3 predict_captchas_with_model.py