import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def map_label(label):
    # Map benign class
    if 'Benign' in label:
        return 'benign'
    
    # Map Spoofing classes
    elif 'ARP_Spoofing' in label:
        return 'ARP Spoofing'
    
    # Map Recon classes
    elif 'Ping_Sweep' in label:
        return 'Ping Sweep'
    elif 'Recon-VulScan' in label:
        return 'Recon VulScan'
    elif 'OS_Scan' in label:
        return 'OS Scan'
    elif 'Port_Scan' in label:
        return 'Port Scan'
    
    # Map MQTT classes
    elif 'Malformed_Data' in label:
        return 'Malformed Data'
    elif 'MQTT-DoS-Connect_Flood' in label:# or 'MQTT-DDoS-Connect_Flood' in label
        return 'DoS Connect Flood'
    elif 'MQTT-DoS-Publish_Flood' in label:
        return 'DoS Publish Flood'
    elif 'MQTT-DDoS-Publish_Flood' in label:
        return 'DDoS Publish Flood'
    elif 'MQTT-DDoS-Connect_Flood' in label:
        return 'DDoS Connect Flood'
    
    # Map DoS classes
    elif 'TCP_IP-DoS-TCP' in label:
        return 'DoS TCP'
    elif 'TCP_IP-DoS-ICMP' in label:
        return 'DoS ICMP'
    elif 'TCP_IP-DoS-SYN' in label:
        return 'DoS SYN'
    elif 'TCP_IP-DoS-UDP' in label:
        return 'DoS UDP'
    
    # Map DDoS classes
    elif 'TCP_IP-DDoS-SYN' in label:
        return 'DDoS SYN'
    elif 'TCP_IP-DDoS-TCP' in label:
        return 'DDoS TCP'
    elif 'TCP_IP-DDoS-ICMP' in label:
        return 'DDoS ICMP'
    elif 'TCP_IP-DDoS-UDP' in label:
        return 'DDoS UDP'
    
    # Return unknown if not classified
    return 'unknown'

def sixlabel(label):
    if label=='benign':
        return 'benign'
    elif label == 'ARP Spoofing':
        return 'Spoofing'
    elif label == 'Ping Sweep' or label == 'Recon VulScan' or label == 'OS Scan' or label == 'Port Scan':
        return 'Recon'
    elif label == 'Malformed Data' or label == 'DoS Connect Flood' or label == 'DoS Publish Flood' or label == 'DDoS Publish Flood' or label == 'DDoS Connect Flood':
        return 'MQTT'
    elif label == 'DoS TCP' or label == 'DoS ICMP' or label == 'DoS SYN' or label == 'DoS UDP':
        return 'Dos'
    elif label == 'DDoS SYN' or label == 'DDoS TCP' or label == 'DDoS ICMP' or label == 'DDoS UDP':
        return 'DDos'

if __name__ == '__main__':

    df_train = pd.read_csv('./CICIoMT/CIC_IoMT_2024_WiFi_MQTT_train.csv')
    df_test = pd.read_csv('./CICIoMT/CIC_IoMT_2024_WiFi_MQTT_test.csv')
    
    df_train['label'] = df_train['label'].map(map_label)
    df_test['label'] = df_test['label'].map(map_label)
    
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)
    
    feature_cols = df_train.columns.drop(['sixclass','label'])

    scaler = MinMaxScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])
    
    
    # Save the training set
    df_train.to_csv('processed_train.csv', index=False)
    
    # Save the test set
    df_test.to_csv('processed_test.csv', index=False)

























