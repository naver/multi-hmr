using UnityEngine;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

public class JSONReader : MonoBehaviour
{
    public static List<SMPLXParams> ReadJSONFile(string filePath)
    {
        if (File.Exists(filePath))
        {
            string jsonContent = File.ReadAllText(filePath);
            SMPLXParams parameters = JsonConvert.DeserializeObject<SMPLXParams>(jsonContent);
            return new List<SMPLXParams> { parameters };
        }
        else
        {
            Debug.LogError("JSON file not found: " + filePath);
            return null;
        }
    }
}

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
public class CameraParams
{
    public int[] imageSize;
    public float[] focalLength;
}

[System.Serializable]
public class SMPLXParams
{
    public int image_width;
    public int image_height;
    public float[][] camera_intrinsics;
    public HumanParams[] humans;
    public CameraParams camera;
    public float[] locationV;
    public float distance;
    public float[] rotationV;
    public float[] positionOffsetV;
    public float[] rotationOffsetV;
}