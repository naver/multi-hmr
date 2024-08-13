using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

public static class JSONReaderFrames
{
    public static List<FrameData> ReadJSONFile(string filePath)
{
    if (File.Exists(filePath))
    {
        string jsonContent = File.ReadAllText(filePath);
        List<FrameData> frames = JsonConvert.DeserializeObject<List<FrameData>>(jsonContent);
        
        // Verificar el orden de los frames
        for (int i = 0; i < frames.Count; i++)
        {
            if (frames[i].frame_id != i)
            {
                Debug.LogWarning($"Frame order mismatch: Expected frame_id {i}, but got {frames[i].frame_id}");
            }
        }
        
        return frames;
    }
    else
    {
        Debug.LogError("JSON file not found: " + filePath);
        return null;
    }
    }
}

public class FrameData
{
    public int frame_id;
    public int resized_width;
    public int resized_height;
    public int checkpoint_resolution;
    public float[][] camera_intrinsics;
    public List<Human> humans;
}

public class Human

{
    public float[] location;
    public float[] translation;
    public float[][] translation_pelvis;
    public float[][] rotation_vector;
    public float[] expression;
    public float[][] joints_2d;
    public float[][] joints_3d;

}
