using UnityEngine;
using System.IO;
using Newtonsoft.Json;

[System.Serializable]
public class HumanParams
{
    public float[] location;
    public float[] translation;
    public float[][] translation_pelvis;
    public float[][] rotation_vector;
    public float[] expression;
    public float[] shape;
    public float[][] joints_2d;
}

[System.Serializable]
public class SMPLXParams
{
    public int image_width;
    public int image_height;
    public float[][] camera_intrinsics;
    public HumanParams[] humans;
}

public class JSONReader : MonoBehaviour
{
    public static SMPLXParams ReadJSONFile(string filePath)
    {
        if (File.Exists(filePath))
        {
            string jsonContent = File.ReadAllText(filePath);
            SMPLXParams parameters = JsonConvert.DeserializeObject<SMPLXParams>(jsonContent);
            return parameters;
        }
        else
        {
            Debug.LogError("JSON file not found: " + filePath);
            return null;
        }
    }
}