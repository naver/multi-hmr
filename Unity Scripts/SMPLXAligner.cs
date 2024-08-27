using UnityEngine;
using UnityEngine.Video;  // Importar la biblioteca para manejar videos
using System.Collections.Generic;

public class SMPLXAligner : MonoBehaviour
{
    public string jsonFilePath;
    public GameObject smplxPrefab;
    public Camera alignmentCamera;
    public VideoPlayer videoPlayer;  // Referencia al componente VideoPlayer
    public bool showJoints2D = true;
    public float joint2DSize = 5f;
    public Color joint2DColor = Color.green;
    public bool showJoints3D = true;
    public Color joint3DColor = Color.blue;
    public float joint3DScale = 0.02f;
    
    public string smplxLayerName = "Smpl-x";  // Nombre de la capa Smpl-x

    private int smplxLayer;
    private int currentFrameIndex = 0;
    private List<FrameData> allFramesData;
    private FrameData currentFrameData;
    private HumanParams currentHumanParams;
    private GameObject smplxInstance;
    private SMPLX smplxComponent;
    private Vector2[] joints2DScreenPositions;
    private GameObject joint3dParentGO;
    private List<GameObject> jointObjects = new List<GameObject>();

    private static readonly string[] SMPLX_POSE_JOINT_NAMES = new string[]
    {
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", 
        "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", 
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_index1", "left_index2", "left_index3", 
        "left_middle1", "left_middle2", "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", 
        "left_ring1", "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3", 
        "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2", "right_middle3", 
        "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1", "right_ring2", "right_ring3", 
        "right_thumb1", "right_thumb2", "right_thumb3", "jaw"
    };

    private static readonly string[] SMPLX_ALL_JOINT_NAMES = new string[] { "pelvis", "left_hip", "right_hip", "spine1", "left_knee",
     "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar",
      "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_index1", 
      "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", 
      "left_ring1", "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3", "right_index1", "right_index2", 
      "right_index3", "right_middle1", "right_middle2", "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3", 
      "right_ring1", "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3", "jaw", "left_eye_smplhf", 
      "right_eye_smplhf", "nose", "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe", "left_small_toe", "left_heel", 
      "right_big_toe", "right_small_toe", "right_heel", "left_thumb", "left_index", "left_middle", "left_ring", "left_pinky", 
      "right_thumb", "right_index", "right_middle", "right_ring", "right_pinky", "right_eye_brow1", "right_eye_brow2", 
      "right_eye_brow3", "right_eye_brow4", "right_eye_brow5", "left_eye_brow5", "left_eye_brow4", "left_eye_brow3", 
      "left_eye_brow2", "left_eye_brow1", "nose1", "nose2", "nose3", "nose4", "right_nose_2", "right_nose_1", "nose_middle", 
      "left_nose_1", "left_nose_2", "right_eye1", "right_eye2", "right_eye3", "right_eye4", "right_eye5", "right_eye6", "left_eye4", 
      "left_eye3", "left_eye2", "left_eye1", "left_eye6", "left_eye5", "right_mouth_1", "right_mouth_2", "right_mouth_3", 
      "mouth_top", "left_mouth_3", "left_mouth_2", "left_mouth_1", "left_mouth_5", "left_mouth_4", "mouth_bottom", "right_mouth_4", 
      "right_mouth_5", "right_lip_1", "right_lip_2", "lip_top", "left_lip_2", "left_lip_1", "left_lip_3", "lip_bottom", 
      "right_lip_3", "right_contour_1", "right_contour_2", "right_contour_3", "right_contour_4", "right_contour_5", 
      "right_contour_6", "right_contour_7", "right_contour_8", "contour_middle", "left_contour_8", "left_contour_7", 
      "left_contour_6", "left_contour_5", "left_contour_4", "left_contour_3", "left_contour_2", "left_contour_1" };
    void Start()
    {
        // Obtener el valor entero de la capa Smpl-x
        smplxLayer = LayerMask.NameToLayer(smplxLayerName);
        videoPlayer.Pause();
        allFramesData = JSONReader.ReadJSONFile(jsonFilePath);
        if (allFramesData != null && allFramesData.Count > 0)
        {
            currentFrameIndex = 0;
            currentFrameData = allFramesData[currentFrameIndex]; 
            currentHumanParams = currentFrameData.humans[0];

            smplxInstance = Instantiate(smplxPrefab);
            SetLayerRecursively(smplxInstance, smplxLayer);
            //smplxInstance.layer = smplxLayer;  // Asignar la capa Smpl-x
            smplxComponent = smplxInstance.GetComponent<SMPLX>();
            smplxInstance.transform.rotation *= Quaternion.Euler(0, 0, 180);

            joint3dParentGO = new GameObject("Joints3DParent");
            joint3dParentGO.SetActive(showJoints3D);
            joint3dParentGO.layer = smplxLayer;  // Asignar la capa Smpl-x

            SetupCamera(currentFrameData);
            AlignSMPLX(currentHumanParams);
        }
        else
        {
            Debug.LogError("Failed to load JSON data or no frames available.");
        }
    }

    void Update()
    {
        if (joint3dParentGO != null && showJoints3D != joint3dParentGO.activeSelf)
        {
            joint3dParentGO.SetActive(showJoints3D);
        }
    }

    void OnGUI()
    {
        if (joints2DScreenPositions == null) return;

        if (showJoints2D)
        {
            // Crear un estilo para los círculos
            GUIStyle circleStyle = new GUIStyle();
            Texture2D texture = new Texture2D(1, 1);
            texture.SetPixel(0, 0, joint2DColor);
            texture.Apply();
            circleStyle.normal.background = texture;

            // Dibujar cada joint como un círculo rojo
            for (int i = 0; i < joints2DScreenPositions.Length; i++)
            {
                Vector2 pos = joints2DScreenPositions[i];
                
                // Invertir la coordenada Y para la GUI de Unity
                pos.y = Screen.height - pos.y;
                
                Rect rect = new Rect(pos.x - joint2DSize / 2, pos.y - joint2DSize / 2, joint2DSize, joint2DSize);
                GUI.Box(rect, GUIContent.none, circleStyle);
                
                // Opcionalmente, puedes dibujar el índice del joint para depuración
                // GUI.Label(new Rect(pos.x, pos.y, 20, 20), i.ToString());
            }
        }
    }
    
    void SetLayerRecursively(GameObject obj, int newLayer)
    {
        if (obj == null)
        {
            return;
        }

        obj.layer = newLayer;

        foreach (Transform child in obj.transform)
        {
            if (child == null)
            {
                continue;
            }
            SetLayerRecursively(child.gameObject, newLayer);
        }
    }

    private void ApplyFrameDataToSMPLX(int frameIndex)
    {
        if (frameIndex < 0 || frameIndex >= allFramesData.Count)
        {
            Debug.LogWarning("Frame index out of bounds");
            return;
        }

        currentFrameIndex = frameIndex;
        currentFrameData = allFramesData[currentFrameIndex];
        currentHumanParams = currentFrameData.humans[0];

        AlignSMPLX(currentHumanParams);

        // Cambiar el frame del video al nuevo frame
        if (videoPlayer != null)
        {
            videoPlayer.Pause();
            videoPlayer.frame = frameIndex;
            Debug.Log($"Video frame set to: {frameIndex}");
        }
    }

    public void OnNextFrameButtonClicked()
    {
        int nextFrameIndex = (currentFrameIndex + 1) % allFramesData.Count;
        ApplyFrameDataToSMPLX(nextFrameIndex);

        if (currentHumanParams != null && currentHumanParams.translation != null && currentHumanParams.translation_pelvis != null)
        {
            AlignMannequin(smplxInstance, 15, currentHumanParams.translation, 0, currentHumanParams.translation_pelvis[0]);
        }
        else
        {
            Debug.LogWarning("No valid data available to align mannequin.");
        }
    }

    public void OnPreviousFrameButtonClicked()
    {
        int previousFrameIndex = (currentFrameIndex - 1 + allFramesData.Count) % allFramesData.Count;
        ApplyFrameDataToSMPLX(previousFrameIndex);

        if (currentHumanParams != null && currentHumanParams.translation != null && currentHumanParams.translation_pelvis != null)
        {
            AlignMannequin(smplxInstance, 15, currentHumanParams.translation, 0, currentHumanParams.translation_pelvis[0]);
        }
        else
        {
            Debug.LogWarning("No valid data available to align mannequin.");
        }
    }

    private void SetupCamera(FrameData frameData)
    {
        float fx = frameData.camera_intrinsics[0][0];
        float fy = frameData.camera_intrinsics[1][1];
        float cx = frameData.camera_intrinsics[0][2];
        float cy = frameData.camera_intrinsics[1][2];

        float fovY = 2 * Mathf.Atan(frameData.resized_height / (2 * fy)) * Mathf.Rad2Deg;
        alignmentCamera.fieldOfView = fovY;

        float aspect = (float)frameData.resized_width / frameData.resized_height;
        alignmentCamera.aspect = aspect;
    }

    private void AlignSMPLX(HumanParams human)
    {
        if (human.rotation_vector != null) ApplyPose(smplxComponent, human.rotation_vector);

        //smplxInstance.transform.rotation *= Quaternion.Euler(0, 0, 180);

        if (human.translation != null && human.translation_pelvis != null)
        {
            AlignMannequin(smplxInstance, 15, human.translation, 0, human.translation_pelvis[0]);
        }

        if (human.joints_2d != null)
        {
            CalculateJoints2DScreenPositions(human.joints_2d, currentFrameData.resized_width, currentFrameData.resized_height);
        }

        if (human.joints_3d != null)
        {
            CreateJoints3D(human.joints_3d, joint3dParentGO.transform);
        }
    }

    private void ApplyPose(SMPLX smplxMannequin, float[][] newPose)
    {
        for (int i = 0; i < newPose.Length; i++)
        {
            string jointName = SMPLX_POSE_JOINT_NAMES[i];
            float rodX = newPose[i][0];
            float rodY = newPose[i][1];
            float rodZ = newPose[i][2];

            Quaternion rotation = SMPLX.QuatFromRodrigues(rodX, rodY, rodZ);
            smplxMannequin.SetLocalJointRotation(jointName, rotation);
        }

        smplxMannequin.EnablePoseCorrectives(true);
        Debug.Log("Pose applied successfully.");
    }

    private void AlignMannequin(GameObject smplxInstance, int primary3DJointIndex, float[] primary3DJoint, int secondary3DJointIndex, float[] secondary3DJoint)
    {
        AdjustScale(smplxInstance, primary3DJointIndex, primary3DJoint, secondary3DJointIndex, secondary3DJoint);
        AdjustPosition(smplxInstance, primary3DJointIndex, primary3DJoint);
        AdjustRotation(smplxInstance, primary3DJointIndex, primary3DJoint, secondary3DJointIndex, secondary3DJoint);
    }

    private void AdjustScale(GameObject smplxGO, int primary3DJointIndex, float[] primary3DJoint, int secondary3DJointIndex, float[] secondary3DJoint)
    {
        Vector3 primary3DJointRef = new Vector3(primary3DJoint[0], -primary3DJoint[1], primary3DJoint[2]);
        Vector3 secondary3DJointRef = new Vector3(secondary3DJoint[0], -secondary3DJoint[1], secondary3DJoint[2]);

        float targetDistance = Vector3.Distance(primary3DJointRef, secondary3DJointRef);
        Vector3 primary3DJointSmplx = smplxGO.GetComponent<SMPLX>().JointTransforms[SMPLX_ALL_JOINT_NAMES[primary3DJointIndex]].position;
        Vector3 secondary3DJointSmplx = smplxGO.GetComponent<SMPLX>().JointTransforms[SMPLX_ALL_JOINT_NAMES[secondary3DJointIndex]].position;
        float currentDistance = Vector3.Distance(primary3DJointSmplx, secondary3DJointSmplx);
        float scaleFactor = targetDistance / currentDistance;

        smplxGO.transform.localScale *= scaleFactor;
    }

    private void AdjustPosition(GameObject smplxInstance, int primary3DJointIndex, float[] primary3DJoint)
    {
        Transform primary3DJointSmplxTransform = smplxInstance.GetComponent<SMPLX>().JointTransforms[SMPLX_ALL_JOINT_NAMES[primary3DJointIndex]];
        Vector3 relativePositionToPrimaryJoint = primary3DJointSmplxTransform.position - smplxInstance.transform.position;

        Vector3 primary3DJointRef = new Vector3(primary3DJoint[0], -primary3DJoint[1], primary3DJoint[2]);
        smplxInstance.transform.position = primary3DJointRef - relativePositionToPrimaryJoint;
    }

    private void AdjustRotation(GameObject smplxGO, int primary3DJointIndex, float[] primary3DJoint, int secondary3DJointIndex, float[] secondary3DJoint)
    {
        Vector3 primary3DJointSmplx = smplxGO.GetComponent<SMPLX>().JointTransforms[SMPLX_ALL_JOINT_NAMES[primary3DJointIndex]].position;
        Vector3 secondary3DJointSmplx = smplxGO.GetComponent<SMPLX>().JointTransforms[SMPLX_ALL_JOINT_NAMES[secondary3DJointIndex]].position;

        Vector3 primary3DJointRef = new Vector3(primary3DJoint[0], -primary3DJoint[1], primary3DJoint[2]);
        Vector3 secondary3DJointRef = new Vector3(secondary3DJoint[0], -secondary3DJoint[1], secondary3DJoint[2]);

        GameObject smplxPivotGO = new GameObject();
        smplxPivotGO.layer = smplxLayer;  // Asignar la capa Smpl-x
        Transform smplxPivot = smplxPivotGO.transform;
        smplxPivot.position = primary3DJointSmplx;
        smplxGO.transform.parent = smplxPivot;

        Vector3 currentSecondaryJointDirection = (secondary3DJointSmplx - primary3DJointSmplx).normalized;
        Vector3 targetSecondaryJointDirection = (secondary3DJointRef - primary3DJointRef).normalized;

        Quaternion additionalRotation = Quaternion.FromToRotation(currentSecondaryJointDirection, targetSecondaryJointDirection);

        smplxPivot.rotation *= additionalRotation;

        smplxGO.transform.parent = null;

        Destroy(smplxPivotGO);
    }

    private void CalculateJoints2DScreenPositions(float[][] joints2D, int imageWidth, int imageHeight)
    {
        joints2DScreenPositions = new Vector2[joints2D.Length];

        float scaleX = (float)Screen.width / imageWidth;
        float scaleY = (float)Screen.height / imageHeight;

        for (int i = 0; i < joints2D.Length; i++)
        {
            float x = joints2D[i][0];
            float y = joints2D[i][1];
            joints2DScreenPositions[i] = new Vector2(x * scaleX, Screen.height - y * scaleY);
        }
    }

    private void CreateJoints3D(float[][] joints3D, Transform parent)
    {
        if (jointObjects.Count == 0)
        {
            // Crear las esferas solo si no han sido creadas previamente
            int jointCount = Mathf.Min(joints3D.Length, SMPLX_ALL_JOINT_NAMES.Length);
            
            parent.position = new Vector3(joints3D[0][0], -joints3D[0][1], joints3D[0][2]);
            for (int i = 0; i < jointCount; i++)
            {
                float[] joint = joints3D[i];
                GameObject jointObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                jointObject.transform.parent = parent;
                jointObject.transform.position = new Vector3(joint[0], -joint[1], joint[2]);
                jointObject.transform.localScale = Vector3.one * joint3DScale;
                jointObject.layer = smplxLayer;  // Asignar la capa Smpl-x
                jointObjects.Add(jointObject);

                if (i < SMPLX_ALL_JOINT_NAMES.Length)
                {
                    jointObject.name = i + "_" + SMPLX_ALL_JOINT_NAMES[i];
                }
                else
                {
                    jointObject.name = i + "_UnknownJoint";
                }

                Destroy(jointObject.GetComponent<SphereCollider>());

                Renderer sphereRenderer = jointObject.GetComponent<Renderer>();
                if (sphereRenderer != null)
                {
                    sphereRenderer.material.color = joint3DColor;
                }
            }
            Debug.Log("Joints3D created successfully.");
        }
        else
        {
            // Actualizar posiciones de las esferas existentes
            for (int i = 0; i < joints3D.Length && i < jointObjects.Count; i++)
            {
                float[] joint = joints3D[i];
                jointObjects[i].transform.position = new Vector3(joint[0], -joint[1], joint[2]);
            }
            Debug.Log("Joints3D positions updated.");
        }

        AlignWithPelvis2D(parent);
    }

    private void AlignWithPelvis2D(Transform pelvis3D)
    {
        Vector2 pelvis2D = joints2DScreenPositions[0];
        Vector3 newPelvis3DPosition = alignmentCamera.ScreenToWorldPoint(new Vector3(pelvis2D.x, pelvis2D.y, pelvis3D.position.z));
        pelvis3D.position = newPelvis3DPosition;
    }

    public void Setjoint3DScale(float scale)
    {
        joint3DScale = scale;
        foreach (GameObject joint in jointObjects)
        {
            joint.transform.localScale = Vector3.one * joint3DScale;
        }
    }
}

