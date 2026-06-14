###From the 20 files with the top words/SHAP values for retracted (10) and valid (10) papers,
###accumulate all the words in each category. The 20 iterations of mistral31_word_shap4/5.py
###were run in parallel on 20 machines as each took 2.5 days.
import datetime
import re
import csv

def cleanword(word):
    word = str(word).strip().lower()
    word = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", word)  ###strip punctuation at ends
    if len(word) > 1:
        return word
    else:
        return ''

if __name__ == "__main__":
    print(datetime.datetime.now())
    retracted = {}
    valid = {}
    for i in range(10):
        ###with open('retracted_words_shap_values_100test_'+str(i)+'.csv', 'r') as file:
        with open('retracted_words_shap_values_100_'+str(i)+'.csv', 'r') as file:
            for j, line in enumerate(file):
                ###if j % 1000 == 0:
                ###    print(j)
                ###if j > 1:
                ###    raise SystemExit
                line = line.replace('[','').replace(']','').replace("\n",'').replace('"','').replace("'","").replace(' ','')
                if line != '':
                    line = line.split(',')
                    line[0] = cleanword(line[0])
                    if line[0] != '' and line[0] in retracted:
                        ###if line[0] == 'numerous':
                            ###print("line[0] in retracted")
                            ###print('len len(line)-1', len(line)-1)
                            ###for k in range(len(line)):
                            ###    print(k, line[k])
                            ###print("len(line)", len(line),"line[0]", line[0], "rest", [line[k] for k in range(2,len(line)-1)])
                        ###if line[0] == 'numerous':
                        ###    print("retracted before update", retracted[line[0]])
                        [retracted[line[0]].append(float(line[k])) for k in range(2,len(line)-1)]
                        ###if line[0] == 'numerous':
                        ###    print("retracted after update", retracted[line[0]])
                    else:
                        ###if line[0] == 'numerous':
                        ###    print("no, line[0] not in retracted")
                        retracted[line[0]] = [float(line[k]) for k in range(2,len(line)-1)]
                    '''
                    if line[0] == 'numerous':
                        for k in range(len(line)):
                            print(k, line[k])
                        print("len(line)", len(line),"line[0]", line[0], "rest", [line[k] for k in range(2,len(line)-1)])
                        print("retracted\n", retracted[line[0]])
                    '''
        with open('valid_words_shap_values_100_'+str(i)+'.csv', 'r') as file:
            for j, line in enumerate(file):
                ###replace elements of the abstract with nothing so that it doesn't confuse ChatGPT
                line = line.replace('[','').replace(']','').replace("\n",'').replace('"','').replace("'","").replace(' ','')
                if line != '':
                    line = line.split(',')
                    line[0] = cleanword(line[0])
                    ###print("line[0]", line[0], "rest", line[2:-1])
                    if line[0] != '' and line[0] in valid:
                        [valid[line[0]].append(float(line[k])) for k in range(2,len(line)-1)]
                    else:
                        valid[line[0]] = [float(line[k]) for k in range(2,len(line)-1)]
    with open('retracted_words_shap_values_total.csv', 'w') as f:
        writer = csv.writer(f)

        ###print(len(retracted))
        retracted_ave_shap = {}
        for j, (key, value) in enumerate(retracted.items()):
            ###if key == 'numerous':
            ###    print(key, value)
            retracted_ave_shap[key] = sum(value)/len(value)
            writer.writerow([key, retracted_ave_shap[key], value])
            ###if key == 'numerous':
            ###    print(retracted_ave_shap[key])
        ###top_shap_words = sorted(retracted_ave_shap, key=lambda k: abs(retracted_ave_shap[k]), reverse=True)[:20]
    ###top_shap_words = sorted([k for k, v in retracted.items() if len(v) > 0],
    ###                        key=lambda k: abs(retracted_ave_shap[k]),reverse=True)[:40]
    top_shap_words = sorted([k for k, v in retracted.items() if len(v) > 10],
                            key=lambda k: retracted_ave_shap[k],reverse=True)
    ###for word in top_shap_words:
    ###    print(word, len(retracted[word]), retracted_ave_shap[word])
    with open('retracted_words_shap_values_top20.csv', 'w') as f:
        writer = csv.writer(f)
        for word in top_shap_words[:10]:
            print(word, len(retracted[word]), retracted_ave_shap[word])
            writer.writerow([word, retracted_ave_shap[word]])
        for word in top_shap_words[-10:]:
            print(word, len(retracted[word]), retracted_ave_shap[word])
            writer.writerow([word, retracted_ave_shap[word]])
        
    with open('valid_words_shap_values_total.csv', 'w') as f:
        writer = csv.writer(f)
        valid_ave_shap = {}
        for j, (key, value) in enumerate(valid.items()):
            valid_ave_shap[key] = sum(value)/len(value)
            writer.writerow([key, valid_ave_shap[key], value])
    top_shap_words = sorted([k for k, v in valid.items() if len(v) > 10],
                            key=lambda k: valid_ave_shap[k],reverse=True)
    print()
    with open('valid_words_shap_values_top20.csv', 'w') as f:
        writer = csv.writer(f)
        for word in top_shap_words[:10]:
            print(word, len(valid[word]), valid_ave_shap[word])
            writer.writerow([word, valid_ave_shap[word]])
        for word in top_shap_words[-10:]:
            print(word, len(valid[word]), valid_ave_shap[word])
            writer.writerow([word, valid_ave_shap[word]])
        
    print(datetime.datetime.now())
    