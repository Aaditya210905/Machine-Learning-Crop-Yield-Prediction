import pandas as pd
from sklearn.preprocessing import PowerTransformer
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(X_train=None,X_test=None,y_train=None, y_test=None,sample_df=None,is_sample=False,encoder=None, ohe=None, pt=None):
    cat_cols = ['Crop','Season','State']
    num_cols = ['Area','Production','Annual_Rainfall','Fertilizer','Pesticide']
    
    if is_sample:
        sample_df['Season'] = sample_df['Season'].str.strip()
        sample_df['State'] = sample_df['State'].str.strip()
        sample_df['Crop'] = sample_df['Crop'].str.strip()
        # Target Encoding for 'Crop'
        sample_df['Crop'] = encoder.transform(sample_df['Crop'])

        # PowerTransform numerical columns
        sample_df[num_cols] = pt.transform(sample_df[num_cols])

        # OneHotEncoding for 'Season' and 'State'
        sample_ohe = ohe.transform(sample_df[['Season', 'State']])
        sample_ohe_df = pd.DataFrame(sample_ohe, columns=ohe.get_feature_names_out(['Season','State']))

        sample_df.reset_index(drop=True, inplace=True)
        sample_df = pd.concat([sample_df, sample_ohe_df], axis=1)
        sample_df.drop(columns=['Season','State'], inplace=True)

        return sample_df

    else:
        X_train['Season'] = X_train['Season'].str.strip()
        X_train['State'] = X_train['State'].str.strip()
        X_train['Crop'] = X_train['Crop'].str.strip()
        X_train.drop(columns=['Crop_Year'], inplace=True)

        X_test['Season'] = X_test['Season'].str.strip()
        X_test['State'] = X_test['State'].str.strip()
        X_test['Crop'] = X_test['Crop'].str.strip()
        X_test.drop(columns=['Crop_Year'], inplace=True)



        # Target Encoding for 'Crop' column
        encoder = ce.TargetEncoder(cols=['Crop'])
        X_train['Crop'] = encoder.fit_transform(X_train['Crop'], y_train)
        X_test['Crop'] = encoder.transform(X_test['Crop'])

        # Power Transformation for numerical columns
        pt = PowerTransformer(method='yeo-johnson')
        X_train[num_cols] = pt.fit_transform(X_train[num_cols])
        X_test[num_cols] = pt.transform(X_test[num_cols])

        pt_y = PowerTransformer(method='yeo-johnson')
        y_train = pt_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test = pt_y.transform(y_test.values.reshape(-1, 1))

        #one-hot encoding for 'Season' and 'State' columns
        ohe = OneHotEncoder(sparse_output=False)

        X_ohe = ohe.fit(X_train[['Season','State']])
        X_ohe = ohe.transform(X_train[['Season','State']])
        X_ohe_test = ohe.transform(X_test[['Season','State']])
        ohe.get_feature_names_out(['Season','State'])

        X_ohe = pd.DataFrame(X_ohe, columns=ohe.get_feature_names_out(['Season','State']))
        X_ohe_test = pd.DataFrame(X_ohe_test, columns=ohe.get_feature_names_out(['Season','State']))

        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)

        X_train = pd.concat([X_train, X_ohe], axis=1)
        X_test = pd.concat([X_test, X_ohe_test], axis=1)

        X_train.drop(columns=['Season','State'], inplace=True)
        X_test.drop(columns=['Season','State'], inplace=True)

        return X_train, X_test, y_train, y_test, encoder, ohe, pt, pt_y, cat_cols, num_cols