## Files

- the ```/data/input_dataset_reformatting/``` folder contains the cleaned dataset from the teachers and the code relative to its reformatting into the ```roads.csv``` file.

The new formatting of the ```roads.csv``` file is the following :

|     Column | Description                                              |
| ---------: | :------------------------------------------------------- |
|       road | On which road does the component belong to               |
|         id | **Unique ID** of the component                           |
| model_type | Type (i.e. class) of the model component to be generated |
|       name | Name of the object                                       |
|        lat | Latitude in Decimal Degrees                              |
|        lon | Longitude in Decimal Degrees                             |
|     length | Length of the object in meters                           |
|  condition | 0 for condition A, 1 for condition B...                  |
|     lengthR| Length of the R component of a bridge                    |
|     lengthL| Length of the L component of a bridge                    |
|  conditionR| Condition of the R component of a bridge                 |
|  conditionL| Condition of the L component of a bridge                 |
|  bridgedual| 1 if bridge has L&R component, empty otherwise           |

- the ```N1road.csv``` file contains all the truncated part of the N1 road from the ```roads.csv``` file, corresponding to the portion between Chittagong and Dhaka
