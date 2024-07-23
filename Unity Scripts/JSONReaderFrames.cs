using UnityEngine;
using System.IO;
using Newtonsoft.Json;
using System.Collections.Generic; // Necesario para usar List

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
    public float[][] joints_3d;
}

[System.Serializable]
public class FrameData
{
    public int frame_id;
    public int resized_width;
    public int resized_height;
    public int checkpoint_resolution;
    public float[][] camera_intrinsics;
    public List<HumanParams> humans;
}

public static class JSONReaderFrames
{
    public static List<FrameData> ReadJSONFile(string filePath)
    {
        if (File.Exists(filePath))
        {
            string jsonContent = File.ReadAllText(filePath);
            List<FrameData> frames = JsonConvert.DeserializeObject<List<FrameData>>(jsonContent);
            return frames;
        }
        else
        {
            Debug.LogError("JSON file not found: " + filePath);
            return null;
        }
    }
}
