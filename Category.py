
import pandas as pd
import spacy
import re
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

class Category:
    """
    The Category class processes shipment descriptions for category classification.

    Input:
          - Expects a DataFrame with a column named 'goods_description'

    Output:
          - Adds 'final_category_code' column representing the classified category.
    """


    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the Category classifier and subcategory mappings."""
        self._nlp = spacy.load("en_core_web_sm")
        self._sbert = SentenceTransformer(model_name)
        self.classifier = None
        self.category_mapping = None

        # Define subcategories for mechanical parts
        self._subcategory_keywords = {
       'Engine_parts': [
            'engine', 'oil pump', 'water pump', 'fuel pump', 'compressor', 'turbocharger', 'radiator', 'engine block',
            'gearbox', 'crankshaft', 'valve', 'filter', 'cylinder', 'spark plug', 'timing belt',
            'marine engine', 'diesel engine', 'propulsion system',
            'motore', 'pompa olio', 'pompa acqua', 'pompa carburante', 'compressore', 'turbocompressore',
            'radiatore', 'blocco motore', 'cambio', 'albero motore', 'valvola', 'filtro', 'cilindro', 'candela',
            'cinghia di distribuzione'
        ],
        'Ship_spare_parts': [
            'marine pump', 'seawater pump', 'ballast pump', 'engine room', 'ship', 'navigation', 'rudder', 'deck',
            'propeller', 'anchor', 'winch', 'stern tube', 'lifeboat', 'gangway', 'fuel system',
            'cooling system', 'navigation system', 'bilge pump', 'hull', 'keel', 'thruster', 'nautical equipment',
            'timoneria', 'elica', 'pompa marina', 'pompa acqua di mare', 'scafo', 'chiglia' , 'oil sep spare', 'oil sep'
        ],
        'Heavy_equipment_parts': [
            'hydraulic pump', 'valve', 'hoses', 'control system', 'excavator', 'grader', 'backhoe', 'loader',
            'track', 'bucket', 'cylinder', 'blade', 'attachments', 'ripper', 'bearing', 'axle', 'boom', 'arm',
            'roller', 'turbocharger', 'gear', 'control panel', 'brake system', 'hydraulic cylinder',
            'bulldozer', 'crane', 'conveyor belt', 'trenching system',
            'pompa idraulica', 'valvola', 'tubi', 'escavatore', 'livellatrice', 'terna', 'caricatore', 'traccia',
            'secchio', 'cilindro', 'lama',  'ripper', 'cuscinetto', 'asse', 'machinery', 'reconditioned bowl'
        ],
        'Appliances': [
            'refrigerator', 'air conditioning', 'sewing machine', 'refrigerator spare', 'washing machine',
            'compressor', 'cooling system',
            'frigorifero', 'macchina per cucire', 'lavatrice', 'condizionatore d\'aria'
        ],
       'Automotive_parts': [
            'car mechanicals', 'auto part', 'vehicle part', 'rubber suspension', 'classic auto part',
            'spare tire', 'automotive',
            'pezzo auto', 'pezzi di veicolo', 'sospensione'
        ],
       'Tools_and_Instruments': [
            'mechanical tool', 'toolbox', 'screwdriver', 'abrasive tool', 'torch', 'impact wrench', 'measuring instrument',
            'drill', 'grinder', 'spanner', 'chisel', 'pliers', 'hammer', 'plastic fitting',
            'utensile', 'trapano', 'metro a nastro', 'attrezzo', 'cacciavite', 'mechanicals tool',
            'mech tool', 'fuel kit', 'empty fuel kit', 'dry bulb'
        ],
       'Fragile_and_Delicate_items': [
            'glassware', 'ceramics', 'crystal', 'porcelain', 'fragile items',
            'vetro', 'ceramica', 'cristallo', 'porcellana', 'fragile'
        ],
       'Lost_and_Found': [
            'lost', 'found', 'unclaimed', 'misplaced', 'left behind', 'abandoned',
            'recovered', 'forgotten', 'baggage', 'luggage', 'property',
            'unclaimed baggage', 'unclaimed luggage', 'found property', 'lost item'
        ],
       'Spare_Parts': [
            'spare part', 'spare parts', 'pump spare part', 'pump spare parts', 'mech spare part',
            'mechanical spare parts', 'mechanical spare part', 'mechanicals spare part', 'mechanical part',
            'mechanicals spare', 'mechanical spare', 'mech spare'
        ],
        'Electronics_and_Gadgets': [

            'bobine','coil','electric','elettrico','tablet','telefono','phone','telecomando','remote control','batteria',
            'battery','luci','lights','trasduttori pressione','pressure transducer','cellulare','mobile phone','caricatore',
            'charger','smartphone','cuffie','headphones','auricolari','earphones','monitor','schermo','screen','display',
            'altoparlante','speaker','amplificatore','amplifier','microfono','microphone','orologio intelligente',
            'smartwatch','fotocamera','camera','portatile','laptop','computer','pc','tastiera','keyboard','mouse',
            'power bank','router','modem','drone','gadget','fitness tracker','gps','usb','adattatore','adapter',
            'cavo caricatore','charger cable','dispositivo bluetooth','bluetooth device','cavo hdmi','hdmi cable',
            'scheda circuito','circuit board','semiconduttore','semiconductor','scheda madre','motherboard',
            'scheda grafica','graphics card','processore','processor','dispositivo intelligente','smart device',
            'oscilloscopio','oscilloscope','multimetro','multimeter','generatore di segnali','signal generator',
            'sensori industriali','industrial sensors','controllore logico programmabile',
            'programmable logic controller','plc','pannello di controllo','control panel','display industriale',
            'industrial display','touchscreen industriale','industrial touchscreen','calibratore','calibrator',
            'termometro','thermometer','registratore dati','data logger','trasduttore','transducer','sensore di pressione',
            'pressure sensor','telecamera a infrarossi','infrared camera','sensore a ultrasuoni','ultrasonic sensor',
            'relè industriale','industrial relay','controllore motore servo','servo motor controller','inverter di potenza',
            'power inverter','alimentazione elettrica','power supply','azionamento motore dc','dc motor drive',
            'azionamento motore ac','ac motor drive','variatore di frequenza','variable frequency drive','amplificatore rf',
            'rf amplifier','timer digitale','digital timer','interruttori industriali','industrial switches',
            'trasformatore di corrente','current transformer','trasformatore di tensione','voltage transformer',
            'sistema di automazione industriale','industrial automation system','dispositivo modbus','modbus device',
            'switch ethernet','ethernet switch','strumento di test elettronico','electronic testing device',
            'strumento di calibrazione elettronico','electronic calibration tool', 'electronic material', 'electronic'
        ],

        'Food_and_Beverages': ['vino', 'wine', 'olio oliva', 'olive oil', 'almond oil' ],

        'Hazardous_Materials': ['hexane', 'chemical', 'chemicals' ],

        'Unknown': ['accessori interni', 'overall', 'varie'],

        'General_Mechanical_Parts': ['mechanical part', 'tool', 'accessory' ],

        'Clothing_and_Accessories': [
            'jacket', 'giacca', 'backbag', 'zaini', 'shirt', 'camicia', 'pants', 'pantaloni', 'jeans',
            'skirt', 'gonna', 'dress', 'vestito', 'suit', 'abito', 'coat', 'cappotto', 'sweater', 'maglione',
            'scarf', 'sciarpa', 'hat', 'cappello', 'shoes', 'scarpe', 'boots', 'stivali', 'belt', 'cintura',
            'bag', 'borsa', 'handbag', 'borsetta', 'wallet', 'portafoglio', 'backpack', 'zaino', 'sunglasses',
            'occhiali da sole', 'watch', 'orologio', 'jewelry', 'gioielli', 'necklace', 'collana', 'earrings',
            'orecchini', 'ring', 'anello', 'gloves', 'guanti', 'umbrella', 'ombrello', 'sportswear',
            'abbigliamento sportivo', 'swimsuit', 'costume da bagno', 'blouse', 'camicetta', 't-shirt', 'maglietta'
        ]

        }


    def _process_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess text descriptions."""
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english')).union(set(stopwords.words('italian')))

        def clean_text(text: str) -> str:
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
            return " ".join(text)

        df['Good_description_clean'] = df['goods_description'].astype(str).apply(clean_text)
        return df

    def _preprocess_and_assign_unknown(self, df, column_name, category_column):
        """Handle missing values and assign 'Unknown' if the row is empty."""
        df[column_name] = df[column_name].fillna('').str.strip()
        df.loc[df[column_name] == '', column_name] = 'empty_row'
        df.loc[df[column_name] == 'empty_row', category_column] = 'Unknown'

    def classify_mechanical_subcategory(self, description):
        """Assigns a specific mechanical subcategory if applicable."""
        description = description.lower()
        for subcategory, keywords in self._subcategory_keywords.items():
            if any(keyword in description for keyword in keywords):
                return subcategory
        return 'General_Mechanical_Parts'

    def train_classifier(self, labeled_df: pd.DataFrame):
        """Train the category classifier using labeled data."""
        if 'Good_description_clean' not in labeled_df.columns:
            labeled_df = self._process_text_fields(labeled_df)
            labeled_df = labeled_df.rename(columns={'Good_description_clean': 'text'})
        else:
            labeled_df = labeled_df.rename(columns={'Good_description_clean': 'text'})

        labeled_df = labeled_df.rename(columns={'matched_category': 'label'})

        # Handle missing values before training
        self._preprocess_and_assign_unknown(labeled_df, 'text', 'label')

        labeled_df['label'] = labeled_df['label'].astype('category')
        labeled_df['label_code'] = labeled_df['label'].cat.codes  # Assign numerical codes to categories

        X_train, X_val, y_train, y_val = train_test_split(
            labeled_df['text'], labeled_df['label_code'], test_size=0.2, random_state=42
        )
        X_train_embeddings = self._sbert.encode(X_train.tolist(), show_progress_bar=True)
        X_val_embeddings = self._sbert.encode(X_val.tolist(), show_progress_bar=True)

        self.classifier = RandomForestClassifier(random_state=42)
        self.classifier.fit(X_train_embeddings, y_train)

        y_pred = self.classifier.predict(X_val_embeddings)
        print("\n Training Completed! Classification Report:\n")
        print(classification_report(y_val, y_pred))

        self.category_mapping = dict(enumerate(labeled_df['label'].cat.categories))

    def categorize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorizes shipments based on their descriptions and assigns subcategories."""

        # Ensure text preprocessing only for raw shipment data
        if 'Good_description_clean' not in df.columns:
          df = self._process_text_fields(df)

        # Handle empty rows before categorization
        self._preprocess_and_assign_unknown(df, 'Good_description_clean', 'predicted_category')

        X_unlabeled_embeddings = self._sbert.encode(df['Good_description_clean'].tolist(), show_progress_bar=True)

        # Assign primary categories
        df['predicted_category_code'] = self.classifier.predict(X_unlabeled_embeddings)
        df['predicted_category'] = df['predicted_category_code'].map(self.category_mapping)

        # Assign subcategories based on keywords
        def assign_subcategory(row):
            if row['predicted_category'] in ["Engine_parts", "Ship_spare_parts", "Automotive_parts"
                                             "Heavy_equipment_parts", "Tools_and_Instruments"]:
                return self.classify_mechanical_subcategory(row['Good_description_clean'])
            for subcategory, keywords in self._subcategory_keywords.items():
                if any(keyword in row['Good_description_clean'] for keyword in keywords):
                    return subcategory
            return "Unknown"

        df['subcategory'] = df.apply(assign_subcategory, axis=1)


        # Create `final_category` column (Use subcategory if available, otherwise main category)
        df['final_category'] = df.apply(
             lambda row: row['subcategory'] if row['subcategory'] != "Unknown" else row['predicted_category'], axis=1
        )

        # Generate a mapping for final categories
        unique_final_categories = df['final_category'].unique()
        final_category_mapping = {category: idx for idx, category in enumerate(unique_final_categories)}
        # Assign `final_category_code` based on final category mapping
        df['final_category_code'] = df['final_category'].map(final_category_mapping)

        # Drop intermediate columns, keep only final_category_code
        columns_to_drop = [
            'Good_description_clean',
            'predicted_category',
            'predicted_category_code',
            'subcategory',
            'final_category',
            'goods_description'
        ]
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        return df
