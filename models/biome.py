import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
import pickle
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
import matplotlib

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_data(csv_path):
    """
    Load molecular dataset with SMILES strings and target properties
    """
    print("Loading dataset...")
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded with {len(df)} molecules")
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def calculate_descriptors(smiles_list):
    """
    Calculate RDKit descriptors for a list of SMILES strings
    """
    descriptors = []
    valid_smiles = []
    
    print("Calculating molecular descriptors...")
    for smiles in tqdm(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Calculate 2D descriptors
                descr = {}
                descr['MolWt'] = Descriptors.MolWt(mol)
                descr['LogP'] = Descriptors.MolLogP(mol)
                descr['NumHDonors'] = Descriptors.NumHDonors(mol)
                descr['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
                descr['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
                descr['NumAromaticRings'] = Lipinski.NumAromaticRings(mol)
                descr['TPSA'] = Descriptors.TPSA(mol)
                descr['FractionCSP3'] = Descriptors.FractionCSP3(mol)
                
                descriptors.append(descr)
                valid_smiles.append(smiles)
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {str(e)}")
            continue
    
    if not descriptors:
        print("Warning: No valid descriptors were generated!")
        return pd.DataFrame(), []
        
    return pd.DataFrame(descriptors), valid_smiles

def generate_new_molecule(template_mol):
    """Generate a new molecule by modifying the template"""
    try:
        # Basic validity check
        if template_mol is None:
            return None, ["Invalid template molecule"]

        # Convert template to SMILES and back to ensure it's valid
        template_smiles = Chem.MolToSmiles(template_mol)
        mol = Chem.MolFromSmiles(template_smiles)
        
        if mol is None:
            return None, ["Invalid molecule"]

        # List of simpler modifications that are less likely to fail
        modifications = [
            'replace_atom',
            'add_group'
        ]
        
        # Apply random modifications
        n_modifications = random.randint(1, 2)  # Reduce number of modifications
        modification_history = []
        
        for _ in range(n_modifications):
            modification = random.choice(modifications)
            
            if modification == 'add_group':
                # Simpler functional groups
                groups = ['O', 'N', 'F', 'Cl']
                group = random.choice(groups)
                try:
                    mol = AllChem.ReplaceSubstructs(mol, Chem.MolFromSmiles('[H]'), 
                                                  Chem.MolFromSmiles(group))[0]
                    modification_history.append(f"Added {group} group")
                except:
                    continue
                    
            elif modification == 'replace_atom':
                try:
                    atoms = list(range(mol.GetNumAtoms()))
                    atom_idx = random.choice(atoms)
                    new_atoms = ['C', 'N', 'O']  # Simpler atom types
                    new_atom = random.choice(new_atoms)
                    editable_mol = Chem.EditableMol(mol)
                    editable_mol.ReplaceAtom(atom_idx, Chem.Atom(new_atom))
                    mol = editable_mol.GetMol()
                    modification_history.append(f"Replaced atom at position {atom_idx} with {new_atom}")
                except:
                    continue

        # Sanitize and return
        try:
            Chem.SanitizeMol(mol)
            return mol, modification_history
        except:
            return None, ["Sanitization failed"]
            
    except Exception as e:
        return None, [f"Error: {str(e)}"]

class BiomimeticSMILESGenerator:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.explainer = None
        self.template_mols = []
        self.generation_history = []
        matplotlib.use('Agg')  # Set non-interactive backend
        plt.ioff()  # Turn off interactive mode
        
    def save_model(self, filepath="biomimetic_model.pkl",scaler_filepath="scaler.pkl"):
        """
        Save the trained model and scaler to a pickle file.
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'template_mols': [Chem.MolToSmiles(mol) for mol in self.template_mols] # Save SMILES
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self, filepath="biomimetic_model.pkl", scaler_filepath="scaler.pkl"):
        """
        Load a trained model from a pickle file.
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.template_mols = [Chem.MolFromSmiles(s) for s in model_data['template_mols']]
            print(f"Model loaded successfully from {filepath}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {filepath}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")

    def train(self, X, y, feature_names, template_smiles):
        """
        Train the model and store template molecules
        """
        self.X_train = X
        self.y_train = y
        self.feature_names = feature_names
        
        # Store template molecules
        self.template_mols = [Chem.MolFromSmiles(s) for s in template_smiles if Chem.MolFromSmiles(s)]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print("Training biomimetic model...")
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=None
        )
        self.model.fit(X_train, y_train)
        
        # Initialize LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            mode="regression",
            random_state=None
        )
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Model R² on train: {train_score:.3f}, test: {test_score:.3f}")
        
        # Explain model's global behavior
        self.explain_model_global()
        
        return test_score
    
    def explain_model_global(self):
        """
        Provide global explanation of the model
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Get feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 6))
        plt.title('Global Feature Importances in Property Prediction')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Print feature importance summary
        print("\nGlobal Feature Importance Summary:")
        for idx in indices:
            print(f"{self.feature_names[idx]}: {importances[idx]:.4f}")
    
    def explain_prediction_local(self, X_instance, prediction):
        """
        Provide local explanation for a specific prediction
        """
        if self.explainer is None:
            raise ValueError("Model must be trained first")
        
        # Get LIME explanation
        exp = self.explainer.explain_instance(
            X_instance, 
            self.model.predict,
            num_features=len(self.feature_names)
        )
        
        # Plot local explanation
        plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure()
        plt.title(f'Local Explanation for Prediction: {prediction:.2f}')
        plt.tight_layout()
        plt.show()
        
        # Return feature contributions
        return exp.as_list()
    
    def generate_smiles(self, target_property_value, n_attempts=1, show_plots=False):
        """
        Generate new SMILES with desired property
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        best_mol = None
        best_prediction = None
        best_distance = float('inf')
        best_template = None
        best_modifications = None
        
        print("\nGenerating new molecules...")
        for _ in tqdm(range(n_attempts)):
            # Select random template
            template_mol = random.choice(self.template_mols)
            template_smiles = Chem.MolToSmiles(template_mol)
            
            # Generate new molecule
            new_mol, modifications = generate_new_molecule(template_mol)
            
            if new_mol is None:
                continue
                
            # Calculate descriptors
            try:
                descr = {}
                descr['MolWt'] = Descriptors.MolWt(new_mol)
                descr['LogP'] = Descriptors.MolLogP(new_mol)
                descr['NumHDonors'] = Descriptors.NumHDonors(new_mol)
                descr['NumHAcceptors'] = Descriptors.NumHAcceptors(new_mol)
                descr['NumRotatableBonds'] = Descriptors.NumRotatableBonds(new_mol)
                descr['NumAromaticRings'] = Lipinski.NumAromaticRings(new_mol)
                descr['TPSA'] = Descriptors.TPSA(new_mol)
                descr['FractionCSP3'] = Descriptors.FractionCSP3(new_mol)
                
                X = pd.DataFrame([descr])
                X_scaled = self.scaler.transform(X)
                prediction = self.model.predict(X_scaled)[0]
                
                distance = abs(prediction - target_property_value)
                
                if distance < best_distance:
                    best_distance = distance
                    best_mol = new_mol
                    best_prediction = prediction
                    best_template = template_smiles
                    best_modifications = modifications
                    
            except Exception as e:
                continue
        
        if best_mol is not None:
            generated_smiles = Chem.MolToSmiles(best_mol)
            
            # Store generation information
            generation_info = {
                'template': best_template,
                'generated': generated_smiles,
                'target_value': target_property_value,
                'predicted_value': best_prediction,
                'modifications': best_modifications
            }
            self.generation_history.append(generation_info)
            
            # Explain the generation
            self.explain_generation(generation_info)
            
            if show_plots:
                # Plot generation process
                plt.figure(figsize=(10, 5))
                # ... plotting code ...
                plt.savefig('generation_plot.png')
                plt.close()
            
            return generated_smiles, best_prediction
        return None, None
    
    def explain_generation(self, generation_info):
        """
        Explain the generation process for a molecule
        """
        template_mol = Chem.MolFromSmiles(generation_info['template'])
        generated_mol = Chem.MolFromSmiles(generation_info['generated'])
        
        # Plot molecules side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        img1 = Draw.MolToImage(template_mol)
        ax1.imshow(img1)
        ax1.set_title("Template Molecule")
        ax1.axis('off')
        
        img2 = Draw.MolToImage(generated_mol)
        ax2.imshow(img2)
        ax2.set_title("Generated Molecule")
        ax2.axis('off')
        
        plt.suptitle("Molecule Generation Process")
        plt.tight_layout()
        plt.show()
        
        # Print modification steps
        print("\nGeneration Process:")
        print(f"Target property value: {generation_info['target_value']:.2f}")
        print(f"Achieved property value: {generation_info['predicted_value']:.2f}")
        print("\nModification steps:")
        for i, mod in enumerate(generation_info['modifications'], 1):
            print(f"{i}. {mod}")

    def get_lime_explanation(self, X):
        """
        Get LIME explanation for the model's predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if self.explainer is None:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                X,
                feature_names=self.feature_names,
                mode="regression"
            )
        
        # Create explanation plot
        exp = self.explainer.explain_instance(
            X[0], 
            self.model.predict,
            num_features=10
        )
        
        # Save plot
        plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure()
        return plt

def main():
    try:
        csv_path = "coconut1.csv"
        
        # Load data
        df = load_data(csv_path)
        print("Available columns:", df.columns.tolist())
        
        if 'SMILES' not in df.columns:
            print("Error: SMILES column not found in the dataset!")
            return
        
        # Create synthetic activity values
        print("Creating synthetic activity values...")
        df['activity'] = df['SMILES'].apply(lambda x: 
            Descriptors.ExactMolWt(Chem.MolFromSmiles(x)) / 100.0 
            if Chem.MolFromSmiles(x) is not None else None
        )
        
        # Calculate descriptors
        descriptors_df, valid_smiles = calculate_descriptors(df['SMILES'])
        
        if len(descriptors_df) == 0:
            print("Error: No valid molecules found!")
            return
            
        print(f"Successfully processed {len(valid_smiles)} valid molecules")
        
        # Match valid SMILES with target property
        target_values = []
        for smiles in valid_smiles:
            target_values.append(df[df['SMILES'] == smiles]['activity'].values[0])
        
        # Train model with explanations
        generator = BiomimeticSMILESGenerator()
        generator.train(
            descriptors_df.values, 
            np.array(target_values),
            descriptors_df.columns,
            valid_smiles
        )
        
        # Save the trained model
        generator.save_model("biomimetic_model.pkl")
        
        # Generate multiple new molecules
        print("\nGenerating multiple new molecules...")
        target_value = np.mean(target_values)
        
        generated_molecules = []
        for i in range(5):  # Generate 5 different molecules
            print(f"\nGenerating molecule {i+1}/5...")
            best_smile, predicted_value = generator.generate_smiles(target_value)
            if best_smile is not None:
                generated_molecules.append((best_smile, predicted_value))
                
                # Get local explanation for this prediction
                mol = Chem.MolFromSmiles(best_smile)
                if mol is not None:
                    descr = {}
                    descr['MolWt'] = Descriptors.MolWt(mol)
                    descr['LogP'] = Descriptors.MolLogP(mol)
                    descr['NumHDonors'] = Descriptors.NumHDonors(mol)
                    descr['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
                    descr['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
                    descr['NumAromaticRings'] = Lipinski.NumAromaticRings(mol)
                    descr['TPSA'] = Descriptors.TPSA(mol)
                    descr['FractionCSP3'] = Descriptors.FractionCSP3(mol)
                    
                    X = pd.DataFrame([descr])
                    X_scaled = generator.scaler.transform(X)
                    print("\nLocal explanation for this prediction:")
                    feature_contributions = generator.explain_prediction_local(X_scaled[0], predicted_value)
                    
                    # Print feature contributions
                    print("\nFeature contributions to prediction:")
                    for feature, contribution in feature_contributions:
                        print(f"{feature}: {contribution:+.4f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()