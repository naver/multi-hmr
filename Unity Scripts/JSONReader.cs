using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

public class JSONReader
{
    public class Data
    {
        public List<List<float>> pose { get; set; }
        public List<float> shape { get; set; }
        public List<float> expression { get; set; }
        public List<float> location { get; set; }
        public float distance { get; set; }
        public List<float> rotation { get; set; }
        public List<float> positionOffset { get; set; }
        public List<float> rotationOffset { get; set; }
        public Camera camera { get; set; }
        public List<List<float>> joints2D { get; set; }
    }

    public class Camera
    {
        public List<float> focalLength { get; set; }
        public List<float> principalPoint { get; set; }
        public List<int> imageSize { get; set; }
    }

    private List<Data> dataList;

    public void ReadJSON(string filePath)
    {
        string json = File.ReadAllText(filePath);
        dataList = JsonConvert.DeserializeObject<List<Data>>(json);
    }

    public List<Data> GetData()
    {
        return dataList;
    }
}
