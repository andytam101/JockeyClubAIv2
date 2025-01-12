# stuff to analyse on:
# 1. number of horses per race distribution
# 2. win odds of winners of all races distribution
# 3. win odds of predicted winners
# 4. win odds of true positive

# from database import init_engine, get_session, Race
# from model import load_model
# from model.model_prediction import ModelPrediction
# from evaluate_strategy import simulate_upcoming_race, is_solo
# from utils.pools import *
#
# from datetime import datetime
# from tqdm import tqdm
#
#
# def main():
#     init_engine()
#
#     model = load_model("RankingNN")
#     model_dir = "trained_models/ranking_nn_epoch_10000"
#     predictor = ModelPrediction(model, model_dir)
#
#     session = get_session()
#
#     races = session.query(Race).filter(Race.date >= datetime(2024, 9, 1).date()).all()
#
#     true_positives = []
#     false_positives = []
#     false_negatives = []
#
#     for race in tqdm(races, desc="Running..."):
#         data = simulate_upcoming_race(race)
#         results = predictor.guess_outcome_of_race(session, data)
#         all_ps = results["ALL"]
#         win_result = results[WIN]
#         target_winning = list(filter(lambda x: x.pool == WIN, race.winnings))[0]
#
#         if is_solo(target_winning.combination, win_result[0]):
#             # true positive
#             true_positives.append((float(target_winning.amount), win_result[1]))
#         else:
#             # false
#             winner_probability = all_ps[int(target_winning.combination)]
#             false_negatives.append((float(target_winning.amount), winner_probability))
#             false_participation = list(filter(lambda x: x.number == win_result[0], race.participations))[0]
#             false_positives.append((float(false_participation.win_odds * 10), win_result[1]))
#
#
#     assert len(false_positives) == len(false_negatives)
#     session.close()
#
#     print(true_positives)
#     print(false_positives)
#     print(false_negatives)
#
#
# if __name__ == "__main__":
#     main()

def squared_p_ev(p, win_odds):
    return p * p * win_odds


def display_header(header: str):
    header_text = str.center(header, 50, '=')
    print()
    print(header_text)


def remove_bets(threshold=33.3):
    corresponding = list(zip(false_positives, false_negatives))
    new_true_positives  = list(filter(lambda x: x[0] > threshold, true_positives))
    new_false = list(filter(lambda x: x[0][0] > threshold, corresponding))

    new_false_positives = list(map(lambda x: x[0], new_false))
    new_false_negatives = list(map(lambda x: x[1], new_false))

    return new_true_positives, new_false_positives, new_false_negatives,

true_positives = [(23.5, 3.8005144596099854), (18.0, 3.7390875816345215), (22.5, 3.7390875816345215),
                  (55.0, 4.0487260818481445), (13.0, 3.7390875816345215), (57.5, 3.855210304260254),
                  (24.0, 3.7390875816345215), (20.5, 3.7390875816345215), (17.0, 3.7390875816345215),
                  (33.5, 3.9921860694885254), (24.0, 3.7800939083099365), (45.0, 3.7390875816345215),
                  (28.0, 3.7390875816345215), (52.5, 3.7390875816345215), (31.5, 3.7390875816345215),
                  (42.0, 3.7390875816345215), (15.5, 3.7390875816345215), (34.0, 3.751288652420044),
                  (50.5, 4.318289756774902), (21.0, 3.7390875816345215), (26.0, 3.7390875816345215),
                  (24.0, 3.7390875816345215), (37.5, 3.9345340728759766), (28.0, 3.7390875816345215),
                  (16.0, 3.7390875816345215), (22.0, 3.7926602363586426), (25.5, 3.751016139984131),
                  (26.0, 3.7390875816345215), (21.0, 3.7390875816345215), (24.5, 3.7390875816345215),
                  (31.5, 3.7445931434631348), (14.5, 3.7390875816345215), (23.0, 3.7390875816345215),
                  (42.5, 4.121555328369141), (26.5, 3.7390875816345215), (23.5, 3.7390875816345215),
                  (27.5, 3.7795944213867188), (34.5, 4.920063495635986), (26.0, 3.7390875816345215),
                  (21.0, 3.765774965286255), (36.5, 4.340897083282471), (28.0, 3.7390875816345215),
                  (19.0, 3.7390875816345215), (52.0, 4.743243217468262), (32.5, 4.473850727081299),
                  (37.5, 3.7390875816345215), (13.5, 3.7390875816345215), (15.5, 3.7390875816345215),
                  (24.5, 3.7390875816345215), (29.0, 3.7390875816345215), (27.0, 3.7390875816345215),
                  (40.5, 3.7390875816345215), (16.0, 3.7390875816345215), (36.0, 3.7390875816345215),
                  (31.0, 3.9195711612701416), (36.5, 4.028347969055176), (24.0, 3.7390875816345215),
                  (29.5, 3.7390875816345215), (23.0, 3.7390875816345215), (25.0, 3.752073049545288),
                  (28.0, 3.7390875816345215), (38.0, 3.943389892578125), (28.0, 3.7390875816345215),
                  (24.5, 3.7390875816345215), (20.0, 3.7390875816345215), (18.0, 3.7390875816345215),
                  (11.5, 3.7390875816345215), (11.0, 3.7390875816345215), (36.0, 4.192376136779785),
                  (21.0, 3.7390875816345215), (44.0, 4.106058120727539), (40.0, 4.019690990447998),
                  (48.0, 4.810911655426025), (45.5, 4.177159786224365), (30.0, 3.744130849838257),
                  (39.5, 3.7390875816345215), (45.0, 3.7390875816345215), (33.0, 3.903878927230835),
                  (36.5, 3.9429612159729004), (28.0, 3.7390875816345215), (38.5, 4.1062798500061035),
                  (26.0, 3.7390875816345215), (11.0, 3.7390875816345215), (26.0, 3.7390875816345215),
                  (11.0, 3.7390875816345215), (28.5, 3.7390875816345215), (18.5, 3.7390875816345215),
                  (29.0, 3.7665140628814697), (33.0, 3.7434301376342773), (41.0, 3.9021103382110596),
                  (37.0, 3.9040207862854004), (23.0, 3.7390875816345215), (51.0, 4.4769487380981445),
                  (32.0, 3.918940544128418), (45.5, 3.769465923309326), (24.0, 3.7390875816345215),
                  (28.0, 3.7390875816345215), (32.5, 3.7955799102783203), (24.0, 3.7485508918762207),
                  (18.5, 3.7390875816345215), (30.0, 3.7390875816345215), (22.0, 3.7390875816345215),
                  (27.5, 3.7390875816345215), (30.5, 3.8717567920684814), (28.5, 3.7390875816345215)]
false_positives = [(35.0, 3.7390875816345215), (31.0, 4.00115966796875), (45.0, 4.606727123260498),
                   (36.0, 3.8734488487243652), (100.0, 3.7979648113250732), (36.0, 3.7390875816345215),
                   (27.0, 3.7390875816345215), (39.0, 3.7390875816345215), (53.0, 4.132401943206787),
                   (37.0, 4.008969306945801), (25.0, 3.7390875816345215), (18.0, 3.7390875816345215),
                   (18.0, 3.7390875816345215), (67.0, 4.126015663146973), (31.0, 3.773836612701416),
                   (36.0, 3.7938895225524902), (35.0, 4.148705959320068), (27.0, 3.7390875816345215),
                   (50.0, 3.786363363265991), (50.0, 3.904040813446045), (30.0, 3.7542355060577393),
                   (31.0, 3.7390875816345215), (31.0, 3.7390875816345215), (77.0, 4.058217525482178),
                   (16.0, 3.7390875816345215), (30.0, 5.107171535491943), (29.0, 3.7390875816345215),
                   (19.0, 3.7415924072265625), (30.0, 3.7390875816345215), (26.0, 3.8073887825012207),
                   (33.0, 3.8742575645446777), (40.0, 4.098412990570068), (18.0, 3.7390875816345215),
                   (27.0, 3.7390875816345215), (42.0, 4.5168561935424805), (61.0, 3.8522145748138428),
                   (19.0, 3.7390875816345215), (41.0, 4.607267379760742), (78.0, 4.26853084564209),
                   (15.0, 3.7390875816345215), (39.0, 4.323254108428955), (63.0, 4.06744384765625),
                   (38.0, 3.8032948970794678), (49.0, 3.7390875816345215), (25.0, 3.7390875816345215),
                   (21.0, 3.7390875816345215), (66.0, 4.20541524887085), (46.0, 3.9322967529296875),
                   (19.0, 3.7390875816345215), (31.0, 3.7635765075683594), (17.0, 3.7390875816345215),
                   (19.0, 3.7390875816345215), (64.0, 4.394858360290527), (52.0, 4.049315452575684),
                   (36.0, 3.918196678161621), (53.0, 3.8643581867218018), (52.0, 3.7782516479492188),
                   (23.0, 3.8185360431671143), (33.0, 3.7390875816345215), (39.0, 3.7923717498779297),
                   (32.0, 4.042877197265625), (31.0, 3.860753297805786), (27.0, 3.7390875816345215),
                   (16.0, 3.7390875816345215), (19.0, 3.7390875816345215), (30.0, 3.7390875816345215),
                   (63.0, 4.204594612121582), (21.0, 3.7390875816345215), (34.0, 4.185240268707275),
                   (34.0, 3.813227891921997), (33.0, 4.190232753753662), (24.0, 3.763286590576172),
                   (43.0, 4.192034721374512), (29.0, 3.8281455039978027), (34.0, 3.8218159675598145),
                   (46.0, 4.093541145324707), (54.0, 4.596516132354736), (16.0, 3.7390875816345215),
                   (60.0, 4.4951043128967285), (49.0, 4.914128303527832), (21.0, 3.7390875816345215),
                   (33.0, 3.8214170932769775), (32.0, 3.7482056617736816), (37.0, 4.070628643035889),
                   (25.0, 3.7390875816345215), (47.0, 3.935324192047119), (41.0, 3.7390875816345215),
                   (26.0, 3.7390875816345215), (38.0, 3.873924493789673), (86.0, 3.9266774654388428),
                   (50.0, 4.43394136428833), (35.0, 3.921976327896118), (19.0, 3.7390875816345215),
                   (35.0, 3.7390875816345215), (59.0, 4.747337818145752), (48.0, 3.9180965423583984),
                   (40.0, 4.1335601806640625), (22.0, 3.7390875816345215), (35.0, 3.8558719158172607),
                   (24.0, 3.7390875816345215), (48.0, 4.1262311935424805), (29.0, 4.6462531089782715),
                   (39.0, 3.951843023300171), (41.0, 4.023632049560547), (18.0, 3.7390875816345215),
                   (29.0, 3.7390875816345215), (40.0, 4.643108367919922), (25.0, 3.7390875816345215),
                   (33.0, 4.05271053314209), (26.0, 3.759984016418457), (55.0, 4.38872766494751),
                   (14.0, 3.7390875816345215), (36.0, 3.7390875816345215), (32.0, 3.7390875816345215),
                   (29.0, 3.7740862369537354), (30.0, 3.7495174407958984), (46.0, 3.8464584350585938),
                   (17.0, 3.7390875816345215), (22.0, 3.7390875816345215), (42.0, 4.192912578582764),
                   (38.0, 4.316162586212158), (17.0, 3.7390875816345215), (27.0, 3.7390875816345215),
                   (18.0, 3.7390875816345215), (29.0, 3.834794759750366), (40.0, 4.0710344314575195),
                   (46.0, 4.232379913330078), (37.0, 3.9396228790283203), (28.0, 3.7390875816345215),
                   (22.0, 3.7390875816345215), (25.0, 3.7390875816345215), (25.0, 3.7390875816345215),
                   (41.0, 3.7390875816345215), (34.0, 3.7390875816345215), (20.0, 3.7390875816345215),
                   (26.0, 3.7935609817504883), (40.0, 3.7390875816345215), (49.0, 4.488880157470703),
                   (42.0, 3.9122090339660645), (27.0, 3.7792110443115234), (41.0, 3.9760377407073975),
                   (38.0, 3.8386542797088623), (37.0, 4.149075031280518), (37.0, 4.4427008628845215),
                   (57.0, 3.7390875816345215), (28.0, 3.7390875816345215), (32.0, 3.846036672592163),
                   (43.0, 3.7936999797821045), (39.0, 3.7390875816345215), (31.0, 3.827740430831909),
                   (39.0, 3.801520824432373), (29.0, 3.7390875816345215), (42.0, 3.967566967010498),
                   (37.0, 4.8880205154418945), (28.0, 3.8868091106414795), (40.0, 4.083795070648193),
                   (24.0, 3.7390875816345215), (36.0, 3.7430129051208496), (34.0, 3.7390875816345215),
                   (32.0, 3.7390875816345215), (32.0, 4.5346903800964355), (28.0, 3.7390875816345215),
                   (38.0, 3.78581166267395), (17.0, 3.7390875816345215), (24.0, 3.7390875816345215),
                   (40.0, 4.421176910400391), (28.0, 3.8782689571380615), (43.0, 3.7390875816345215),
                   (35.0, 3.818857192993164), (42.0, 4.50735330581665), (22.0, 3.7390875816345215),
                   (34.0, 3.7390875816345215), (30.0, 3.8823585510253906), (13.0, 3.7390875816345215),
                   (35.0, 3.7390875816345215), (52.0, 4.071456432342529), (32.0, 3.7701313495635986),
                   (25.0, 3.7390875816345215), (23.0, 3.7390875816345215), (34.0, 3.7594168186187744),
                   (25.0, 3.775409460067749), (31.0, 3.7863385677337646), (37.0, 3.7390875816345215),
                   (27.0, 3.7390875816345215), (31.0, 3.7390875816345215), (36.0, 3.7851245403289795),
                   (29.0, 3.7636892795562744), (62.0, 5.0661115646362305), (33.0, 3.7390875816345215),
                   (29.0, 3.7390875816345215), (24.0, 3.7390875816345215), (43.0, 3.919621467590332),
                   (29.0, 3.815051794052124), (55.0, 4.127511501312256), (37.0, 3.766510009765625),
                   (33.0, 3.9928908348083496), (23.0, 3.7472052574157715), (26.0, 3.7390875816345215),
                   (28.0, 3.7390875816345215), (34.0, 3.8819992542266846), (19.0, 3.7390875816345215)]
false_negatives = [(28.0, 3.9412825107574463), (153.5, 5.352252960205078), (188.0, 5.488373279571533),
                   (58.5, 3.9719700813293457), (576.5, 6.617715835571289), (99.0, 4.662658214569092),
                   (56.5, 4.2223944664001465), (36.5, 3.964144468307495), (94.0, 6.572727680206299),
                   (91.0, 6.791341781616211), (43.0, 5.001166820526123), (171.5, 6.727680206298828),
                   (414.0, 6.450644493103027), (159.0, 7.520565986633301), (52.5, 4.359637260437012),
                   (75.5, 4.821499824523926), (59.0, 5.645751476287842), (52.5, 3.82170033454895),
                   (98.5, 7.275173664093018), (79.0, 6.246929168701172), (56.0, 4.352966785430908),
                   (97.5, 6.324747562408447), (188.0, 5.727214813232422), (55.0, 4.363813400268555),
                   (179.5, 4.8815460205078125), (48.5, 6.205083847045898), (49.0, 4.306413650512695),
                   (59.5, 3.836385726928711), (225.0, 6.431499481201172), (89.0, 4.794408798217773),
                   (61.5, 4.2759294509887695), (123.0, 5.0050368309021), (117.0, 5.270916938781738),
                   (94.0, 4.513484001159668), (43.0, 5.13222599029541), (52.5, 4.106723308563232),
                   (58.5, 4.946751117706299), (110.5, 6.166825294494629), (364.5, 7.340160369873047),
                   (58.0, 4.183474540710449), (145.5, 7.574969291687012), (61.5, 4.465154647827148),
                   (154.0, 4.863365173339844), (91.0, 5.139426231384277), (124.0, 8.550250053405762),
                   (215.5, 5.948072910308838), (49.5, 5.764285087585449), (61.0, 4.156627178192139),
                   (104.0, 5.893385887145996), (37.5, 4.2447733879089355), (158.0, 7.4534592628479),
                   (66.0, 4.378391265869141), (141.0, 7.310422420501709), (77.5, 4.206642150878906),
                   (95.0, 5.162891864776611), (31.5, 4.106380462646484), (34.5, 4.416452407836914),
                   (159.0, 6.63066291809082), (57.5, 4.045657157897949), (83.0, 5.655220985412598),
                   (295.0, 7.157132148742676), (82.0, 5.675100326538086), (139.0, 6.594000339508057),
                   (87.5, 5.432560443878174), (121.5, 5.166617393493652), (133.5, 7.382991313934326),
                   (31.5, 4.25642728805542), (215.5, 7.318707466125488), (207.0, 7.366239547729492),
                   (63.0, 4.246461391448975), (138.5, 7.749995708465576), (48.5, 3.837266206741333),
                   (71.5, 5.699834823608398), (433.5, 7.5136823654174805), (159.5, 6.665874004364014),
                   (91.0, 6.914731979370117), (67.0, 4.954403877258301), (73.5, 5.642581939697266),
                   (42.0, 4.639082908630371), (108.5, 7.671900749206543), (49.0, 4.369711875915527),
                   (54.0, 4.313282489776611), (56.5, 4.677237033843994), (164.5, 7.292868137359619),
                   (428.0, 8.609363555908203), (51.5, 4.655557632446289), (49.0, 5.170005798339844),
                   (136.0, 7.290493011474609), (100.5, 7.019679069519043), (44.0, 5.29173469543457),
                   (131.5, 8.121051788330078), (41.0, 4.416260719299316), (82.0, 6.365619659423828),
                   (53.0, 4.870680332183838), (62.5, 5.300272464752197), (73.5, 7.646142959594727),
                   (51.5, 5.281478404998779), (62.5, 5.917070388793945), (105.0, 7.767287731170654),
                   (117.5, 4.97910213470459), (71.0, 6.014897346496582), (91.0, 4.861622333526611),
                   (105.0, 6.086599826812744), (89.0, 6.227106094360352), (104.0, 5.836038112640381),
                   (122.0, 7.38316535949707), (106.5, 6.699605941772461), (81.0, 5.67852258682251),
                   (100.5, 6.863029479980469), (128.5, 5.500348091125488), (56.0, 5.288387298583984),
                   (40.5, 3.9124772548675537), (68.0, 5.448495864868164), (109.0, 6.249093532562256),
                   (41.0, 4.50394344329834), (37.0, 4.007665634155273), (46.0, 4.24626350402832),
                   (74.0, 5.189985275268555), (160.5, 6.1662797927856445), (400.0, 9.055486679077148),
                   (53.0, 4.651719093322754), (35.5, 4.435504913330078), (708.5, 9.124326705932617),
                   (179.5, 9.317842483520508), (120.5, 7.484592437744141), (66.5, 5.713134765625),
                   (106.5, 6.942452430725098), (63.5, 4.524003982543945), (345.0, 8.898368835449219),
                   (265.0, 7.446242332458496), (148.0, 6.854335308074951), (435.0, 8.560159683227539),
                   (122.0, 6.502922058105469), (42.5, 4.548691272735596), (451.5, 8.226890563964844),
                   (69.5, 4.696864128112793), (64.5, 3.961256265640259), (74.0, 4.959585666656494),
                   (169.5, 5.810264587402344), (71.5, 5.9944257736206055), (50.0, 4.278229713439941),
                   (68.0, 5.073558807373047), (48.0, 4.875382423400879), (65.5, 4.539771556854248),
                   (40.0, 3.918574571609497), (50.5, 5.287477016448975), (62.5, 4.687092304229736),
                   (84.5, 5.247600555419922), (37.5, 3.846067428588867), (29.5, 3.8412892818450928),
                   (148.0, 6.036066055297852), (47.5, 4.76622200012207), (384.0, 7.1543684005737305),
                   (100.5, 5.550693988800049), (63.0, 5.216282844543457), (68.5, 6.058470249176025),
                   (35.5, 3.7895267009735107), (76.0, 4.604824542999268), (84.0, 5.283673286437988),
                   (59.5, 5.642266273498535), (122.5, 5.123624324798584), (48.0, 4.311337471008301),
                   (119.5, 4.890622138977051), (121.0, 5.966280937194824), (34.5, 3.9177019596099854),
                   (57.0, 4.875348091125488), (968.5, 9.103004455566406), (58.0, 3.814375877380371),
                   (77.5, 4.6768903732299805), (66.5, 4.8823161125183105), (142.5, 6.975386619567871),
                   (56.0, 5.050724029541016), (56.5, 5.0157318115234375), (67.5, 4.416318893432617),
                   (337.5, 9.549006462097168), (133.0, 5.813612937927246), (62.5, 5.493710517883301),
                   (56.0, 5.487354278564453), (62.0, 5.104310035705566), (96.5, 6.289746284484863),
                   (448.5, 8.93588638305664), (148.0, 7.677228927612305), (72.0, 6.054487228393555),
                   (54.0, 5.750369071960449), (344.5, 8.110342025756836), (95.0, 4.352477550506592),
                   (55.0, 4.078360080718994), (74.0, 5.343557357788086), (135.0, 5.891335964202881),
                   (34.0, 4.199542045593262), (170.0, 5.33233118057251), (198.5, 6.366074085235596),
                   (64.0, 5.860905647277832), (41.5, 4.301183223724365), (132.0, 6.889100074768066),
                   (104.0, 5.768702507019043), (57.5, 4.261227607727051), (183.5, 7.019877910614014),
                   (184.0, 7.28714656829834), (98.5, 6.2240190505981445), (208.0, 8.240558624267578)]

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics


true_positives_win_odds = list(map(lambda x: x[0], true_positives))
false_positives_win_odds = list(map(lambda x: x[0], false_positives))
false_negatives_win_odds = list(map(lambda x: x[0], false_negatives))

true_positives_probability = list(map(lambda x: x[1], true_positives))
false_positives_probability = list(map(lambda x: x[1], false_positives))
false_negatives_probability = list(map(lambda x: x[1], false_negatives))

print(f"Accuracy: {len(true_positives) / (len(true_positives) + len(false_positives))}")
print(
    f"Profit change: {sum(true_positives_win_odds) - (len(false_positives_win_odds) + len(true_positives_win_odds)) * 10}")

true_positives_ev = list(map(lambda x: x[0] * x[1], true_positives))
false_positives_ev = list(map(lambda x: x[0] * x[1], false_positives))
false_negatives_ev = list(map(lambda x: x[0] * x[1], false_negatives))

true_positives_squared_p_ev = list(map(lambda x: squared_p_ev(x[1], x[0]), true_positives))
false_positives_squared_p_ev = list(map(lambda x: squared_p_ev(x[1], x[0]), false_positives))
false_negative_squared_p_ev = list(map(lambda x: squared_p_ev(x[1], x[0]), false_negatives))

display_header("Probabilities")
print(statistics.mean(true_positives_probability))
print(statistics.mean(false_positives_probability))
print(statistics.mean(false_negatives_probability))
display_header("Expected Values")
print(statistics.mean(true_positives_ev))
print(statistics.mean(false_positives_ev))
print(statistics.mean(false_negatives_ev))
display_header("Squared probability EVs")
print(statistics.mean(true_positives_squared_p_ev))
print(statistics.mean(false_positives_squared_p_ev))
print(statistics.mean(false_negative_squared_p_ev))


a, b, c = remove_bets()
print(f"Modified profit: {sum(map(lambda x: x[0], a)) - (len(a) + len(b)) * 10}")


def remove_anomalies(xs, lower_threshold, upper_threshold):
    return list(filter(lambda x : lower_threshold < x < upper_threshold, xs))


true_positives_win_odds = remove_anomalies(true_positives_win_odds, 0, 250)
false_positives_win_odds = remove_anomalies(false_positives_win_odds, 0, 250)
false_negatives_win_odds = remove_anomalies(false_negatives_win_odds, 0, 250)


# sns.histplot(true_positives_win_odds, binwidth=5, kde=True, color="blue", label="True Positive", stat="density")
# sns.histplot(false_positives_win_odds, binwidth=5, kde=True, color="red", label="False Positive", stat="density")
# sns.histplot(false_negatives_win_odds, binwidth=5, kde=True, color="green", label="False Negative", stat="density")


sns.histplot(true_positives_win_odds + false_positives_win_odds, kde=True, color='green',  label="Predicted Distribution", stat="density", binwidth=5)
sns.histplot(true_positives_win_odds + false_negatives_win_odds, kde=True, color='blue', label="Actual Distribution", stat="density", binwidth=5)


def softmax(x):
    return np.exp(-x) / np.sum(np.exp(-x))


# Add labels and title
plt.title('Win Odds Distribution', fontsize=16)
plt.xlabel('Win Odds', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
