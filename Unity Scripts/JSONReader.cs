using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

public class JSONReader
{
    public class Human
    {
        public List<float> location { get; set; }
        public List<float> translation { get; set; }
        public List<List<float>> translation_pelvis { get; set; }
        public List<List<float>> rotation_vector { get; set; }
        public List<float> expression { get; set; }
        public List<float> shape { get; set; }
        public List<List<float>> joints_2d { get; set; }
        public List<List<float>> joints_3d { get; set; }
    }

    public class Data
    {
        public int image_width { get; set; }
        public int image_height { get; set; }
        public int resized_width { get; set; }
        public int resized_height { get; set; }
        public int checkpoint_resolution { get; set; }
        public List<List<float>> camera_intrinsics { get; set; }
        public List<Human> humans { get; set; }
    }

    private Data data;

    public void ReadJSON(string filePath)
    {
        string json = File.ReadAllText(filePath);
        data = JsonConvert.DeserializeObject<Data>(json);
    }

    public Data GetData()
    {
        return data;
    }
}
