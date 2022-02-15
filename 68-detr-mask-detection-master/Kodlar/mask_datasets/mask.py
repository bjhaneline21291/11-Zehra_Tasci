"""
Builds a dataset for 527030_966454_bundle_archive
"""
#%% Setup
import random

from glob             import glob
from xml.etree        import ElementTree
from PIL              import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, ConcatDataset
from torch            import FloatTensor, LongTensor, stack

from utils import cx_cy_to_min_max, min_max_to_cx_cy

#%% Dataset class
class ImageDataset(Dataset):
    """
    Creates a dataset for hardhat and head detection problems
    """
    def __init__( self
                , images     :str
                , annotation_folder:str
                , label_map        :dict
                , transform        :object
                ):
        """
        :return: Dataset. A Dataset object that can be iterated to provide batched samples for training
        """
        # Dataset attributes
        self.images    = images
        self.transform = transform

        # Process labels
        self.object_list = [self.parse_annotation( annotation=f'''{annotation_folder}/{'.'.join(image.split("/")[-1].split('.')[:-1])}.xml'''
                                                 , label_map=label_map
                                                 )
                            for image in [image.replace("\\", '/') for image in self.images]
                           ]


    def __getitem__(self, i):
        # Open and convert an image
        image = Image.open(fp=self.images[i], mode='r').convert('RGB')

        # Get the boxes and labels
        boxes = FloatTensor(self.object_list[i]["boxes"])
        classes = LongTensor(self.object_list[i]["classes"])

        # apply transformations
        image, boxes, classes = self.transform(image, boxes, classes)

        return image, boxes, classes

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        """
        Function for indicating how components of the dataset should be collated
        This is necessary since the boxes and labels will be different sizes for different images
        :param batch: an iterable of N sets that results from calls to __getitem__()
        :return: a tensor of images, and lists of varying-sized tensors containing the labels and boxes
        """
        # Storage
        images = []
        boxes  = []
        classes = []

        # Iterate through the batch and append to storage
        _ = [storage.append(sample[i]) for sample in batch for storage, i in zip((images, boxes, classes), (0,1,2))]

        # Stack the images
        images = stack(images, dim=0)

        return [images, [{"boxes":boxes_, "labels":labels_} for boxes_, labels_ in zip(boxes, classes)]]

    def parse_annotation(self, annotation, label_map):
        """
        Converts an annotation from the hardhat dataset to produce the boxes, classes, and difficulties
        :param annotation: str. Path to a .xml annotation file.
        :param label_map: dict. A dictionary that encodes string labels to a numeric value
        :return: list
        """
        # Get the xml root
        root = ElementTree.parse(source=annotation).getroot()

        # Data storage
        boxes = []
        classes = []
        difficulties = []

        # iterate through the objects in the file
        for obj in root.iter('object'):
            # Get the difficulty
            difficulty = int(obj.find('difficult').text == '1')

            # Get the label
            label = obj.find('name').text.lower().strip()

            # Check if the label is in the label map
            if label not in label_map:
                continue

            # Bounding box coordinates
            bbox = obj.find('bndbox')
            coords = [int(bbox.find(coord + pos).text) - 1 for pos in ('min', 'max') for coord in ('x', 'y')]

            # Convert
            coords = min_max_to_cx_cy(coords)

            # Appending to storage
            _ = [storage.append(item) for storage, item in
                 zip((boxes, classes, difficulties), (coords, label_map[label], difficulty))
                ]

        # return the results
        return {'boxes': boxes, 'classes': classes, 'difficulties': difficulties}

    @staticmethod
    def draw_boxes(image, labels, label_map, colour_map):
        # Copy the image for annotation
        annotated_image = image

        # Get the classes
        inv_label_map = {v: k for k, v in label_map.items()}
        classes = [inv_label_map[class_] for class_ in labels["classes"].tolist()]

        # Convert the boxes to original locations
        image_dims = FloatTensor([image.width, image.height, image.height, image.width])
        boxes = [box * image_dims for box in labels["boxes"]]

        # create a draw and font object
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype("./calibril.ttf", 15)

        for i in range(len(boxes)):
            # Draw boxes
            box = boxes[i].tolist()

            # Convert
            box = cx_cy_to_min_max(box)

            draw.rectangle(xy=box, outline=colour_map[classes[i]])
            #draw.rectangle(xy=[coord + 1 for coord in box], outline='#e6194b')

            # Label text
            size = font.getsize(classes[i].upper())
            loc = [box[0] + 2.0, box[1] - size[1]]
            box = [box[0], box[1] - size[1], box[0] + size[0] + 4.0, box[1]]
            draw.rectangle(xy=box, fill=colour_map[classes[i]])
            draw.text(xy=loc, text=classes[i].upper(), fill="white", font=font)

        # Free memory
        del draw

        return annotated_image

#%%% Test
def view_dataset_batch():
    # Setup
    from torch.utils.data import DataLoader

    import torch
    import torchvision.transforms.functional as F
    import transforms                        as T

    # Create the label maps
    label_map_1 = {"good": 0, "none": 1, "bad": 1, "background": 2}
    label_map_2 = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 1, "background": 2}
    label_map_3 = {"mask": 0, "unmask": 1, "mask_weared_incorrect": 1, "background": 2}

    # Image files
    images_1 = glob("dataset/Face Mask/527030_966454_bundle_archive/images/*")
    images_2 = glob("dataset/Face Mask/667889_1176415_bundle_archive/images/*")
    images_3 = glob("dataset/Face Mask/Maskdata/JPEGImages/*")

    # Create a transform
    transform = T.Compose([T.NormalizeBoxes(), T.Resize([400, 400]), T.ColourJitter(),
                           T.Mirror(), T.ToTensor(), T.RandomNoise()])

    # Create the datasets
    dataset = ConcatDataset(
        [ImageDataset(images=images_1, annotation_folder="dataset/Face Mask/527030_966454_bundle_archive/labels",
                      label_map=label_map_1, transform=transform),
         ImageDataset(images=images_2, annotation_folder="dataset/Face Mask/667889_1176415_bundle_archive/annotations",
                      label_map=label_map_2, transform=transform),
         ImageDataset(images=images_3, annotation_folder="dataset/Face Mask/Maskdata/Annotations",
                      label_map=label_map_3, transform=transform)])

    # Create a dataloader
    loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, collate_fn=ImageDataset.collate_fn)

    # Fetch a batch
    for sample in loader:
        break

    images = sample[0]
    labels = sample[1]

    label_map = {"mask": 0, "with_mask": 0, "good": 0,
                 "no mask": 1, "none": 1, "unmask": 1,
                 "bad": 1, "mask_weared_incorrect": 1, "background": 2}
    colour_map = {key: "#%06x" % random.randint(0, 0xFFFFFF) for key in label_map}

    # Create an empty tensor
    result = torch.zeros([3,800,800])
    for image, labels_, (x_1, x_2, y_1, y_2) in zip(images, labels, [[0,400,0,400],[400,800,0,400],[0,400,400,800],[400,800,400,800]]):
        annotated_image = ImageDataset.draw_boxes( image=F.to_pil_image(image)
                                                 , labels={"boxes":labels_["boxes"], "classes":labels_["labels"]}
                                                 , label_map=label_map_1
                                                 , colour_map=colour_map)
        result[:,y_1:y_2,x_1:x_2] = F.to_tensor(annotated_image)

    result = F.to_pil_image(result)
    result.show()

'''
for i in range(20):
    view_dataset_batch()
    a=0
'''
