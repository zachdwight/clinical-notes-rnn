#ifndef SYNTHETIC_TRAINING_DATA_H
#define SYNTHETIC_TRAINING_DATA_H

#include <vector>
#include <string>
#include <utility> // Required for std::pair

// This data is artificial for demonstration purposes only.
// It does NOT contain real patient information or Protected Health Information (PHI).
// It is designed to mimic clinical notes for a simplified "diagnosis" task.

const std::vector<std::pair<std::string, std::string>> SIMULATED_CLINICAL_NOTES = {
    // --- Cold Cases (approx. 35) ---
    {"Patient presents with mild cough, runny nose, and sore throat. No fever. Symptoms started two days ago.", "Cold"},
    {"Complains of stuffy nose and sneezing. Occasional dry cough. Feeling slightly tired.", "Cold"},
    {"Sore throat, congestion, and low energy. Reports no fever or body aches. Eating well.", "Cold"},
    {"Child has clear nasal discharge and frequent sneezing. Seems otherwise active.", "Cold"},
    {"Mild cough and post-nasal drip. Throat feels scratchy. Denies fever or chills.", "Cold"},
    {"Reports nasal congestion, watery eyes, and a mild headache. Feels a bit under the weather.", "Cold"},
    {"Persistent sneezing episodes. Nasal passages feel blocked. No significant pain.", "Cold"},
    {"Feeling run down with a sore throat that worsens when swallowing. Minimal cough.", "Cold"},
    {"Minor cough producing clear phlegm. Nose is a bit stuffy. Normal temperature.", "Cold"},
    {"Just a general feeling of being unwell, with slight congestion and occasional cough.", "Cold"},
    {"Patient reports a tickle in throat and a slight cough. No fever detected.", "Cold"},
    {"Nasal drip, itchy eyes, and a feeling of general malaise. Denies severe symptoms.", "Cold"},
    {"Mild head cold with sinus pressure. No fever. Able to perform daily activities.", "Cold"},
    {"Aches and pains are minimal. Primary complaint is nasal blockage and sneezing.", "Cold"},
    {"Cough is non-productive. Throat irritation. Eating and drinking fine.", "Cold"},
    {"Developed a stuffy nose and sneezing last night. Feels like a common cold.", "Cold"},
    {"No fever, but has a slight sore throat and runny nose since morning.", "Cold"},
    {"Complains of mild cough, but it's not frequent. Mostly just a sniffle.", "Cold"},
    {"Minor congestion and feeling tired. No other significant complaints.", "Cold"},
    {"Scratchy throat and a bit of a cough. Patient denies flu-like symptoms.", "Cold"},
    {"Runny nose, sneezing, and slight fatigue. Says it's 'just a cold'.", "Cold"},
    {"Headache, but no fever. Feels congestion in sinuses. Normal appetite.", "Cold"},
    {"Symptoms include mild cough, sore throat, and nasal congestion. No difficulty breathing.", "Cold"},
    {"Patient reports feeling unwell with a runny nose and occasional cough.", "Cold"},
    {"Sneezing, mild sore throat. No fever, no muscle aches. Still going to work.", "Cold"},
    {"Ticklish cough, mostly dry. Some nasal discharge. Overall, not too bad.", "Cold"},
    {"Mild chest congestion but denies any breathing issues. Has a cold.", "Cold"},
    {"Woke up with a stuffy head and sneezing. Drinking lots of fluids.", "Cold"},
    {"Symptoms: watery eyes, clear nasal discharge, and a little tired.", "Cold"},
    {"Throat feels raw. Occasional coughing fits but no fever. Normal vitals.", "Cold"},
    {"Slight cough and congestion. Denies chills or body aches. Feeling better than yesterday.", "Cold"},
    {"Nasal congestion and a mild headache. No fever. Symptoms are improving.", "Cold"},
    {"Reports minor sore throat and sneezing. No fever. Denies shortness of breath.", "Cold"},
    {"Just a little rundown, with a stuffy nose. No significant fever or aches.", "Cold"},
    {"Mild cold symptoms. Mostly a runny nose and some throat irritation. Not severe.", "Cold"},


    // --- Flu Cases (approx. 35) ---
    {"Patient presents with sudden onset of high fever, severe muscle aches, and fatigue. Dry cough. Headache.", "Flu"},
    {"Complains of body aches, chills, and a fever of 102F. Feels exhausted. Sore throat present.", "Flu"},
    {"Severe fatigue, headache, and generalized muscle pain. Fever has been intermittent.", "Flu"},
    {"Abrupt onset of symptoms: fever, significant body aches, and dry cough. Denies nasal congestion.", "Flu"},
    {"Chills, sweating, and malaise. Fever is 101.5F. Patient can barely get out of bed.", "Flu"},
    {"Profound fatigue, generalized weakness, and headache. Fever. No runny nose or sneezing.", "Flu"},
    {"Reports severe body aches, sore throat, and a persistent fever. Cough is dry and hacking.", "Flu"},
    {"Myalgia, arthralgia, and high fever. Cough is present. Denies any productive sputum.", "Flu"},
    {"Patient feels like they've been hit by a truck. High fever, chills, and all over body pain.", "Flu"},
    {"Sudden onset of fever and extreme tiredness. Headache is throbbing. Mild cough.", "Flu"},
    {"Fever of 103F. Muscle soreness throughout body. Very weak. Has flu-like symptoms.", "Flu"},
    {"Complains of fever, chills, and severe headache. Dry cough present. No nasal symptoms.", "Flu"},
    {"Body aches and overwhelming fatigue. Temperature 102.8F. Unable to go to work.", "Flu"},
    {"Sudden onset of flu-like illness. Fever, headache, and muscle pain. Non-productive cough.", "Flu"},
    {"Symptoms include high fever, severe body aches, and fatigue. No congestion.", "Flu"},
    {"Patient reports feeling very ill with high fever, chills, and joint pain.", "Flu"},
    {"Generalized weakness, malaise, and a temperature of 102F. Dry, persistent cough.", "Flu"},
    {"Fever, headache, and significant muscle aches. Denies runny nose. Feeling drained.", "Flu"},
    {"Reports acute onset of flu symptoms: fever, chills, and profound fatigue.", "Flu"},
    {"High fever, severe body aches, and headache. Cough is not productive. Feeling very weak.", "Flu"},
    {"Patient has fever, chills, and generalized pain. Difficulty sleeping due to discomfort.", "Flu"},
    {"Symptoms started abruptly: fever, muscle aches, and extreme tiredness. Sore throat.", "Flu"},
    {"Aching muscles, headache, and a high fever. Dry cough is worsening.", "Flu"},
    {"Chills, fever of 101.9F. Patient feels completely wiped out. Some joint pain.", "Flu"},
    {"Complains of sudden fever, severe body aches, and headache. No respiratory issues noted.", "Flu"},
    {"Extreme fatigue and muscle pain. Fever is 102.5F. Unable to eat much.", "Flu"},
    {"Patient reports sudden onset of headache, fever, and severe body aches.", "Flu"},
    {"Fever, chills, and muscle soreness. Cough is dry. Feels very unwell.", "Flu"},
    {"Symptoms: high fever, body aches, and headache. Says it feels like the flu.", "Flu"},
    {"Sudden onset of fever and exhaustion. Generalized muscle pain. No congestion.", "Flu"},
    {"Fever of 102.2F. Severe headache and muscle aches. Dry cough and sore throat.", "Flu"},
    {"Patient presents with classic flu symptoms: high fever, body aches, and profound fatigue.", "Flu"},
    {"Significant malaise, chills, and fever. Muscles ache all over. Cough present.", "Flu"},
    {"Abrupt onset of fever and body pain. Patient feels too sick to work. Headache.", "Flu"},
    {"Flu-like symptoms including fever, muscle pain, and severe fatigue. Denies stuffy nose.", "Flu"},

    // --- Pneumonia Cases (approx. 30) ---
    {"Patient presents with productive cough (green sputum), shortness of breath, and fever. Chest pain with breathing.", "Pneumonia"},
    {"Complains of severe dyspnea, persistent cough with colored phlegm, and fever. Decreased oxygen saturation.", "Pneumonia"},
    {"Fever, chills, and difficulty breathing. Cough is deep and wet, producing yellow sputum. Auscultation reveals crackles.", "Pneumonia"},
    {"Acute onset of fever, cough, and significant shortness of breath. Chest tightness. Fatigue.", "Pneumonia"},
    {"Productive cough with rust-colored sputum. Labored breathing. Fever. Decreased breath sounds at lung base.", "Pneumonia"},
    {"Reports difficulty catching breath, cough with greenish mucus, and fever. Elevated respiratory rate.", "Pneumonia"},
    {"Patient has a persistent cough, fever, and feels winded after minimal exertion. History of lung issues.", "Pneumonia"},
    {"Fever 102F, severe cough producing thick phlegm, and shortness of breath. Rhonchi noted on exam.", "Pneumonia"},
    {"Sudden onset of chest pain, difficulty breathing, and a deep cough with purulent sputum. Weakness.", "Pneumonia"},
    {"Complains of fever, chills, and significant shortness of breath. Crackles heard in lower lobes. Coughing up mucus.", "Pneumonia"},
    {"Labored breathing, cough with thick yellow sputum, and fever. Decreased appetite. Looks unwell.", "Pneumonia"},
    {"Productive cough, high fever, and dyspnea. Oxygen saturation is low. Appears in respiratory distress.", "Pneumonia"},
    {"Patient reports persistent cough with green phlegm, fever, and shortness of breath. Chest X-ray ordered.", "Pneumonia"},
    {"Worsening cough, fever, and feeling very breathless. Coughing up blood-tinged sputum. Weak.", "Pneumonia"},
    {"Severe cough with copious mucus, fever, and difficulty breathing. Rapid, shallow breaths.", "Pneumonia"},
    {"Fever, productive cough, and shortness of breath upon exertion. Some chest discomfort.", "Pneumonia"},
    {"Complains of productive cough, fever, and feeling winded. Reduced breath sounds on one side.", "Pneumonia"},
    {"Patient has a deep cough with purulent discharge, fever, and rapid breathing.", "Pneumonia"},
    {"Fever, chills, and significant shortness of breath. Cough producing yellow-green sputum.", "Pneumonia"},
    {"Persistent cough with thick mucus, fever, and difficulty taking a deep breath.", "Pneumonia"},
    {"Shortness of breath, productive cough, and fever. Feeling very weak and tired.", "Pneumonia"},
    {"Reports fever, cough with green phlegm, and increasing shortness of breath. Needs oxygen.", "Pneumonia"},
    {"Patient presents with cough, fever, and severe breathing difficulties. Suspect pneumonia.", "Pneumonia"},
    {"Wet cough, fever, and dyspnea. Crackling sounds in lungs. Looks pale.", "Pneumonia"},
    {"Difficulty breathing, productive cough, and fever. Unable to lie flat comfortably.", "Pneumonia"},
    {"Coughing up thick yellow sputum, fever, and shortness of breath. Chest pain.", "Pneumonia"},
    {"Fever and cough that produces phlegm. Patient states they are very breathless.", "Pneumonia"},
    {"Deep cough, fever, and gasping for air. Oxygen saturation 89%.", "Pneumonia"},
    {"Patient reports a cough with greenish mucus, fever, and severe breathing problems. Admitted.", "Pneumonia"},
    {"Chest tightness, fever, and a cough producing rust-colored sputum. Dyspneic.", "Pneumonia"}
};

#endif // SIMULATED_DATA_H
