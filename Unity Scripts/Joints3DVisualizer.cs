using System.Collections.Generic;
using UnityEngine;

public class Joints3DVisualizer : MonoBehaviour
{
    public string jsonFilePath;
    public GameObject jointSpherePrefab; // Prefab de la esfera para los joints
    public float jointSphereScale = 0.1f; // Escala de las esferas

    private List<JSONReader.Data> parametersList;

    void Start()
    {
        JSONReader jsonReader = new JSONReader();
        jsonReader.ReadJSON(jsonFilePath);
        parametersList = jsonReader.GetData();

        if (parametersList != null)
        {
            foreach (var parameters in parametersList)
            {
                if (parameters.joints_3d != null)
                {
                    DisplayJoints3D(parameters.joints_3d);
                }
            }
        }
    }

    void DisplayJoints3D(List<List<float>> joints3D)
    {
        foreach (var joint in joints3D)
        {
            if (joint.Count >= 3)
            {
                Vector3 jointPosition = new Vector3(joint[0], joint[1], joint[2]);
                GameObject jointSphere = Instantiate(jointSpherePrefab, jointPosition, Quaternion.identity);
                jointSphere.transform.localScale = Vector3.one * jointSphereScale;
            }
            else
            {
                Debug.LogError("Joint 3D data is not valid.");
            }
        }
    }
}